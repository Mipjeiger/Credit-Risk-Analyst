# Credit Risk — Monthly Report (2026-09)

## Challenger scoreboard
| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   ROC AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Random Forest       |     0.9435 |      0.9446 |   0.9435 |     0.9423 |    0.9673 |
| Decision Tree       |     0.9383 |      0.9406 |   0.9383 |     0.9376 |    0.9453 |
| Gradient Boosting   |     0.9366 |      0.9393 |   0.9366 |     0.9359 |    0.9499 |
| XGBoost             |     0.9302 |      0.9335 |   0.9302 |     0.929  |    0.9408 |
| Logistic Regression |     0.9163 |      0.9177 |   0.9163 |     0.913  |    0.9718 |
| K-Nearest Neighbors |     0.7525 |      0.7515 |   0.7525 |     0.7207 |    0.8877 |

## Monitoring (Gini / KS / PSI)
| Model               |        Gini |         KS |   PSI (train/test) |
|:--------------------|------------:|-----------:|-------------------:|
| K-Nearest Neighbors |  0.211957   | 0.159631   |       12.9825      |
| Gradient Boosting   |  0.177556   | 0.197017   |        0.000824638 |
| Random Forest       |  0.024462   | 0.0409156  |        0.630479    |
| Logistic Regression |  0.00015528 | 0.00015528 |       12.9825      |
| Decision Tree       |  0          | 0          |       12.9825      |
| XGBoost             | -0.188636   | 0.193165   |       12.9825      |

_Generated: 2026-09-17T11:30:42_