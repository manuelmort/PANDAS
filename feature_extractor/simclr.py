import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys
import time

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    # In simclr.py, modify the train() method:

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"])
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model, device_ids=eval(self.config['gpu_ids']))
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        # ========== ADD PROGRESS TRACKING ==========
        total_epochs = self.config['epochs']
        steps_per_epoch = len(train_loader)
        total_steps = total_epochs * steps_per_epoch

        print("\n" + "="*70)
        print("TRAINING CONFIGURATION")
        print("="*70)
        print(f"Total epochs:        {total_epochs}")
        print(f"Steps per epoch:     {steps_per_epoch}")
        print(f"Total steps:         {total_steps}")
        print(f"Batch size:          {self.config['batch_size']}")
        print(f"Log every:           {self.config['log_every_n_steps']} steps")
        print("="*70)
        print()

        start_time = time.time()
        # ===========================================

        for epoch_counter in range(self.config['epochs']):
            # ========== EPOCH START ==========
            epoch_start_time = time.time()
            print(f"\n{'='*70}")
            print(f"EPOCH [{epoch_counter + 1}/{total_epochs}]")
            print(f"{'='*70}")
            # =================================

            for batch_idx, (xis, xjs) in enumerate(train_loader):
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                    # ========== ENHANCED LOGGING ==========
                    elapsed_time = time.time() - start_time
                    steps_remaining = total_steps - n_iter
                    avg_time_per_step = elapsed_time / (n_iter + 1)
                    eta_seconds = steps_remaining * avg_time_per_step
                    eta_minutes = eta_seconds / 60

                    progress_pct = 100 * n_iter / total_steps

                    print(f"[Epoch {epoch_counter + 1}/{total_epochs}] "
                          f"Step {n_iter}/{total_steps} ({progress_pct:.1f}%) | "
                          f"Loss: {loss:.3f} | "
                          f"ETA: {eta_minutes:.1f} min")
                    # ======================================

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # ========== EPOCH END ==========
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch_counter + 1} completed in {epoch_time/60:.2f} minutes")
            # ===============================

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(f"[{epoch_counter + 1}/{total_epochs}] Validation Loss: {valid_loss:.3f}")
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('✓ Model saved (best validation loss)')

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        # ========== TRAINING COMPLETE ==========
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"Final loss: {loss:.3f}")
        print(f"Best validation loss: {best_valid_loss:.3f}")
        print(f"Model saved to: {model_checkpoints_folder}")
        print("="*70)
        # =======================================


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("✓ Loaded pre-trained SimCLR checkpoint with success.")
        except (FileNotFoundError, KeyError):
            print("✓ No SimCLR checkpoint found. Using ImageNet pretrained ResNet backbone.")  # ← Better message

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0

            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
