# Metrics Explanation

This document explains the evaluation metrics used in the deepfake detection pipeline.

---

## 📊 Confusion Matrix

The confusion matrix shows actual vs predicted labels:

```
                 Predicted
              Real    Fake
         ┌────────┬────────┐
Actual   │   TN   │   FP   │   Real
         ├────────┼────────┤
         │   FN   │   TP   │   Fake
         └────────┴────────┘
```

| Term | Meaning | For Deepfake Detection |
|------|---------|------------------------|
| **TN** | Real → predicted Real | ✅ Correct |
| **FP** | Real → predicted Fake | ⚠️ False alarm |
| **FN** | Fake → predicted Real | ❌ **Dangerous** — deepfake spreads |
| **TP** | Fake → predicted Fake | ✅ Correct |

---

## 📈 Key Metrics

### Accuracy
```
Accuracy = (TP + TN) / Total
```
- **Problem:** Misleading with imbalanced data
- **Example:** 80% fake → always predict fake = 80% accuracy

### Precision
```
Precision = TP / (TP + FP)
```
- **Meaning:** "Of all flagged videos, how many were actually fake?"
- **High precision:** Few false alarms

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- **Meaning:** "Of all fake videos, how many did I catch?"
- **High recall:** Catches most fakes

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Meaning:** Harmonic mean of precision and recall
- **Why F1?** Balances catching fakes vs avoiding false alarms

### AUC (Area Under ROC Curve)
- **Meaning:** Probability that a random fake scores higher than a random real
- **Range:** 0.5 (random) to 1.0 (perfect)

---

## ⚠️ Why False Negatives Matter

In deepfake detection, **False Negatives are the dangerous error**:

| Error | Consequence |
|-------|-------------|
| FP (False Alarm) | User reviews video, realizes it's real. Minor inconvenience. |
| FN (Missed Fake) | **Deepfake spreads undetected.** Can cause real harm. |

---

## 🎯 Why I Use F1 Score

| Metric | Problem |
|--------|---------|
| Accuracy | Can be gamed with class imbalance |
| Pure Recall | Would flag everything as fake |
| Pure Precision | Would miss subtle fakes |
| **F1 Score** | ✅ Balances precision and recall |

I use F1 score for:
- Model selection (best model = highest F1)
- Early stopping (stop when F1 plateaus)

---

## 📋 My Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 100% | All predictions correct |
| Precision | 1.00 | No false alarms |
| Recall | 1.00 | All fakes caught |
| F1 Score | 1.00 | Perfect balance |
| AUC | 1.00 | Perfect separation |

**Note:** Perfect scores on 50-video dataset. Real-world generalization requires larger datasets.

---

## 🔧 Threshold Selection

Default threshold is 0.5:
- score > 0.5 → Fake
- score ≤ 0.5 → Real

Can be adjusted for different use cases:

| Threshold | Effect |
|-----------|--------|
| 0.3 | Conservative — catch more fakes, more false alarms |
| 0.5 | Balanced (default) |
| 0.7 | Strict — fewer false alarms, might miss subtle fakes |
