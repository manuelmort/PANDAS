## CSV Files Are Not Uploaded Due to Kaggle's Terms of Service

## Notes for Data Splitting

**Dataset Overview:**
- Total entries in train.csv: **10,616**
- Memory usage: 2.26 MB
- Number of columns: 4

**Sample Data:**
| image_id                          | data_provider | isup_grade | gleason_score |
|-----------------------------------|---------------|------------|---------------|
| 0005f7aaab2800f6170c399693a96917  | karolinska    | 0          | 0+0           |
| 000920ad0b612851f8e01bcc880d9b3d  | karolinska    | 0          | 0+0           |
| 0018ae58b01bdadc8e347995b69f99aa  | radboud       | 4          | 4+4           |
| 001c62abd11fa4b57bf7a6c603a11bb9  | karolinska    | 4          | 4+4           |
| 001d865e65ef5d2579c190a0e0350d8f  | karolinska    | 0          | 0+0           |

---

### Suggested Train/Validation Splits

| Split Ratio | Training Entries | Validation Entries |
|-------------|------------------|-------------------|
| **80/20** ✅ | **8,492 (80%)** | **2,124 (20%)** |
| 85/15       | 9,023 (85%)     | 1,593 (15%)     |
| 90/10       | 9,554 (90%)     | 1,062 (10%)     |

**Recommendation:** We chose the **80/20 split** as it provides a good balance between:
- Sufficient training data (8,492 entries)
- Adequate validation data (2,124 entries) for reliable performance evaluation
- Standard practice in machine learning workflows

---

### Patch Suggestions for Training Data (8,492 entries)

| Patch Size    | Full Patches | Last Patch Size | Total Patches |
|---------------|--------------|-----------------|---------------|
| 1,000 entries | 8            | 492 entries     | 9             |
| 5,000 entries | 1            | 3,492 entries   | 2             |

**Our Approach:** We split the training data into **10 equal patches** (~849 entries each) to:
- Enable sequential processing within memory constraints
- Facilitate incremental data loading from Kaggle to SCC
- Maintain balanced patch sizes for consistent training