# Credit Card Fraud Detection

End-to-end portfolio project demonstrating:
- Handling severe class imbalance (~0.17% fraud)
- Unsupervised anomaly detection (Isolation Forest + PyTorch Autoencoder)
- Supervised classification with imbalance mitigation (XGBoost)
- Model comparison using PR-AUC (most appropriate for imbalance)
- Production-ready FastAPI endpoint with rate limiting

Best model: **XGBoost with class weights**  
→ Final PR-AUC ≈ 0.88+

## Results Summary

| Model                     | PR-AUC   | Key Strength                          | Key Weakness                     |
|---------------------------|----------|----------------------------------------|----------------------------------|
| Isolation Forest          | 0.1717   | Fast, no labels needed                 | Poor precision                   |
| Autoencoder (PyTorch)     | 0.6454   | Excellent anomaly separation (121×)    | Computationally heavier          |
| XGBoost + Class Weights   | 0.8800| Highest PR-AUC, interpretable (SHAP)   | Requires labels                  |

See `reports/figures/final_pr_comparison.png` for visual comparison.

## Model Interpretability

SHAP analysis on the best XGBoost model shows the most important features for fraud detection are:

- V14, V17, V10, V12 (strong negative contribution)
- Amount, V4, V11 (positive contribution in many cases)

![SHAP Summary](reports/figures/shap_summary_fraud.png)