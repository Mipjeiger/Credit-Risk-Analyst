 # Credit Risk Data Science Summary

1. **Purpose**
	- The notebook evaluates credit-risk challenger models and develops supporting model-risk, portfolio, vendor, and automation analyses.

2. **Data and model artifacts**
	- The workflow loads the merged credit-risk dataset from Parquet.
	- Six saved challenger classifiers are loaded from the model artifacts directory:
	  - Logistic Regression
	  - Random Forest
	  - Gradient Boosting
	  - XGBoost
	  - K-Nearest Neighbors
	  - Decision Tree
	- Model metadata, feature names, class labels, metrics, parameters, and confusion matrices are also loaded.

3. **Challenger model evaluation**
	- All six models are rescored on a common holdout dataset.
	- The evaluation calculates weighted Accuracy, Precision, Recall, F1 Score, and multiclass ROC AUC.
	- The scoreboard ranks models by weighted F1 Score and selects the top-ranked model as the champion.
	- The notebook marks this section as **real**, because it uses saved model artifacts and an actual holdout evaluation.

4. **Preprocessing requirement**
	- Categorical values are label-encoded and model inputs are expected to use the same encoding and feature order as training.
	- The saved scaler and label encoders should be used for production inference.
	- The challenger rescoring cells should be checked to ensure they use the saved preprocessing pipeline consistently; fitting a new scaler or encoder during evaluation can invalidate comparisons.

5. **Model monitoring metrics**
	- Gini, KS, and PSI are calculated for the champion and challenger models.
	- The champion output recorded:
	  - Gini: `0.1776`
	  - KS: `0.1970`
	  - PSI, train versus holdout: `0.0008`
	- The configured thresholds are:
	  - Minimum Gini: `0.30`
	  - Minimum KS: `0.20`
	  - Maximum PSI: `0.25`
	- Six model alerts were produced in the local dry run.
	- This is **partial**, because the notebook compares train and holdout data rather than genuine time-separated production populations. True drift monitoring requires scored data collected over time.

6. **Income estimation model**
	- An XGBoost regression model estimates `NETMONTHLYINCOME` while excluding income itself from the features to reduce direct target leakage.
	- Recorded results:
	  - Model MAPE: `11.1123`
	  - Baseline MAPE: `14.0803`
	- The notebook states that an approximately 78% improvement claim cannot be verified without the original benchmark model.
	- This section is **partial** and should be validated with a time-based test set and an agreed business baseline.

7. **Propensity model**
	- An XGBoost classifier predicts a proxy propensity target derived from `last_prod_enq2`.
	- The target treats named product values as positive and `none`, `others`, missing values, and empty strings as negative.
	- Target distribution:
	  - Class 0: `20,831`
	  - Class 1: `30,505`
	- Recorded results:
	  - Average Precision: `0.9010`
	  - ROC AUC: `0.8812`
	  - Positive rate: `59.418%`
	- This is **partial**, because the target is a proxy and may be affected by leakage. A production propensity target should represent a future event, such as an order or application within a defined time window.

8. **Third-party vendor backtest**
	- The notebook compares approval economics with and without vendor information.
	- The current test uses synthetic data and assumed economics:
	  - Vendor fetch cost: `2.5`
	  - Loss per bad approval: `500`
	  - Gain per good approval: `40`
	- Recorded simulated vendor ROI: `21.64x`.
	- This is a **scaffold**, not a production conclusion, because real vendor flags, observed outcomes, costs, and loss assumptions are required.

9. **Portfolio evaluation**
	- The notebook defines calculations for NPF by vintage and FPD30.
	- The current portfolio contains `20,000` synthetic records across 12 monthly vintages.
	- Recorded synthetic portfolio FPD30: `3.00%`.
	- Synthetic NPF by vintage is approximately `5.2%` to `7.1%`.
	- NPF and FPD30 are also grouped by approval band.
	- This is a **scaffold**, because real repayment history, origination vintage, months-on-book, delinquency, and first-payment data are needed.

10. **Automation and reporting**
	 - A monthly Markdown report and CSV metrics file are generated under the `reports` directory.
	 - A SQL query is provided to check recent monitoring records against Gini, KS, and PSI thresholds.
	 - A local dry run identifies model-monitoring breaches from `monitor_df`.
	 - This section is **real for local execution**, but database scheduling, alert delivery, and operational ownership still need to be connected for production use.

11. **Overall conclusion**
	 - The strongest completed capability is challenger model loading, holdout scoring, metrics reporting, and local monitoring calculations.
	 - The monitoring result indicates that the selected thresholds are breached by the current evaluated models, especially where Gini and KS are below minimums or PSI is above the maximum.
	 - Income estimation, propensity modeling, vendor analysis, and portfolio evaluation demonstrate the intended methodology but require production-quality targets and observed business data before they can support decisions.

12. **Recommended next steps**
	 - Re-evaluate every saved model with the exact training-time encoders, scaler, feature order, and target mapping.
	 - Replace holdout PSI with time-windowed production PSI.
	 - Define future-event labels for propensity modeling and remove temporally unavailable features.
	 - Replace synthetic vendor and portfolio data with real operational and repayment data.
	 - Add automated tests for artifact loading, feature compatibility, target labels, metric columns, and monitoring thresholds.

## Summary Insight

The notebook successfully establishes a working credit-risk analytics foundation: six challenger models can be loaded and evaluated, monitoring metrics can be calculated, and automated reports and threshold checks can be generated. However, the current results should be treated as a technical baseline rather than final business evidence. The low champion Gini and KS values, together with monitoring threshold breaches, indicate that model discrimination and stability require further investigation. The income, propensity, vendor, and portfolio analyses are useful prototypes, but their conclusions depend on replacing proxy or synthetic data with time-aligned production outcomes. The next priority is to make preprocessing consistent with training, define reliable future-event targets, and validate all metrics on real temporal and repayment data before using the results for production credit decisions.
