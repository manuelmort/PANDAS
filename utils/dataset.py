"""Dataset class for the graph classification task."""

import os
from warnings import warn
from typing import Any

import torch
from torch.utils import data
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(
        self,
        root: str,
        ids: list[str],
        site: str | None = "LUAD",
        classdict: dict[str, int] | None = None,
        target_patch_size: int | None = None,
    ) -> None:

        super(GraphDataset, self).__init__()
        self.root = root
        self.ids = ids

        # --- Class dictionary handling ---
        if classdict is not None:
            self.classdict = classdict

        else:
            if site is None:
                warn("Neither site nor classdict provided. Assuming labels are integers.")
                self.classdict = None

            elif site in {"LUAD", "LSCC"}:
                self.classdict = {"normal": 0, "luad": 1, "lscc": 2}

            elif site == "NLST":
                self.classdict = {"normal": 0, "tumor": 1}

            elif site == "TCGA":
                self.classdict = {"Normal": 0, "TCGA-LUAD": 1, "TCGA-LUSC": 2}

            elif site == "panda":  
                # PANDA labels are integers 0,1,2 directly
                self.classdict = None

            else:
                raise ValueError(f"Site {site} not recognized and classdict not provided")

        self.site = site

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        info = self.ids[index].replace("\n", "")

        # Expected: "site/ID\tlabel"
        try:
            left, label = info.split("\t")
            site, graph_name = left.split("/")
        except ValueError as exc:
            raise ValueError(
                f"Invalid id format: {info}. Expected 'site/filename\\tlabel'"
            ) from exc

        # Check site consistency
        if self.site is not None:
            if site != self.site:
                return None  # skip samples from other sites

        # Compute expected graph folder
        if site in {"LUAD", "LSCC"}:
            site_key = "LUNG"
            graph_path = os.path.join(self.root, f"CPTAC_{site_key}_features", "simclr_files")

        elif site == "NLST":
            graph_path = os.path.join(self.root, "NLST_Lung_features", "simclr_files")

        elif site == "TCGA":
            graph_path = os.path.join(self.root, "TCGA_LUNG_features", "simclr_files")

        elif site == "panda":
            # PANDA graphs live directly in graphs_all/<id>/
            graph_path = os.path.join(self.root, graph_name)

        else:
            graph_path = os.path.join(self.root, f"{site}_features", "simclr_files")

        # --- Load feature matrix ---
        if site == "panda":
            feature_path = os.path.join(graph_path, "features.pt")
        else:
            feature_path = os.path.join(graph_path, graph_name, "features.pt")

        if not os.path.exists(feature_path):
            return None  # Skip sample instead of crashing

        features = torch.load(feature_path, map_location="cpu")

        # --- Load adjacency ---
        if site == "panda":
            adj_s_path = os.path.join(graph_path, "adj_s.pt")
        else:
            adj_s_path = os.path.join(graph_path, graph_name, "adj_s.pt")

        if not os.path.exists(adj_s_path):
            return None  # Skip sample if missing

        adj_s = torch.load(adj_s_path, map_location="cpu")
        if adj_s.is_sparse:
            adj_s = adj_s.to_dense()

        # --- Create output ---
        sample: dict[str, Any] = {}
        sample["image"] = features
        sample["adj_s"] = adj_s
        sample["label"] = self.classdict[label] if (self.classdict is not None) else int(label)
        sample["id"] = graph_name

        return sample

    def __len__(self):
        return len(self.ids)
