# 💳 Credit Analyst Risk approach by Machine learning engineering fullstack

## 📊 Dataset & Production ML Pipeline

### 1. Data Source

* [ ] Fetch credit-risk data into a Pandas DataFrame.
* [ ] Dataset size: **51,336 records × 87 columns**.
* [ ] Validate schema, data types, missing values, and duplicate records.
* [ ] Use `PROSPECTID` as the unique prospect identifier.
* [ ] Separate numerical and categorical features.

### 2. Feature Engineering

* [ ] Process credit history, delinquency, loan, enquiry, utilization, income, employment, and demographic features.
* [ ] Encode categorical features:

  * `MARITALSTATUS`
  * `EDUCATION`
  * `GENDER`
  * `last_prod_enq2`
  * `first_prod_enq2`
* [ ] Handle special/missing values such as `-99999`.
* [ ] Validate feature distributions and outliers.
* [ ] Prevent data leakage between training and inference.

### 3. Feature Selection

* [ ] Reduce the original **87 columns** to the production feature set.
* [ ] Maintain **47 selected features** based on model importance.
* [ ] Prioritize high-impact credit-risk features such as:

  * `max_recent_level_of_deliq`
  * `pct_tl_open_L6M`
  * `Credit_Score`
  * `PL_utilization`
  * `time_since_recent_payment`
  * `max_deliq_12mts`
  * `num_dbt_12mts`
  * `num_sub_12mts`
  * `num_lss`
  * `Tot_TL_closed_L6M`

### 4. ML Target

* [ ] Use `Approved_Flag` as the prediction target.
* [ ] Build a multiclass credit-approval/risk classification model.
* [ ] Evaluate class distribution and potential imbalance.
* [ ] Track precision, recall, F1-score, confusion matrix, and ROC-AUC where applicable.

## 5. LLM-Based Merchant Risk Decisioning Detection 

* [ ] Integrate an **LLM layer** to support merchant risk detection, investigation, decision explanations, fraud prediction.
* [ ] Combine structured ML predictions to investigate structure risk outputs
    * `fraud risk with flag with explanation`
    * `risk level confidence on policy states within recommend action`
    * `recommendation to handle business problem on critical anomaly detection`
* [ ] Using **RAG** to retrieve relevant policies, risk rules, historical cases, and compliance knowledge before generating conclusions.
* [ ] Develop significant threshold to prevent hallucination
* [ ] Collab with Risk Analyst to maintain reinforce LLM for better output recommendation

### 6. Training Pipeline

* [ ] Build reproducible preprocessing and training pipelines.
* [ ] Train and compare multiple ML algorithms.
* [ ] Train LLM to ampliyfy explanation to answer the reason clarifying problem solution
* [ ] Track experiments and metrics with MLflow.
* [ ] Save preprocessing artifacts and trained models.
* [ ] Register the production model.
* [ ] Version dataset, features, code, and model artifacts.

### 7. Production Inference

* [ ] Fetch prospect features from the database.
* [ ] Apply the same preprocessing used during training.
* [ ] Generate `Approved_Flag` predictions.
* [ ] Return prediction, confidence/probability, risk information, and model version.
* [ ] Expose inference through a production API.
* [ ] Validate incoming feature schema before prediction.

### 8. Monitoring

* [ ] Monitor incoming data quality.
* [ ] Monitor feature drift.
* [ ] Monitor prediction distribution.
* [ ] Monitor model performance after deployment.
* [ ] Track false approvals and false rejections.
* [ ] Monitor API latency, throughput, and errors.
* [ ] Build Prometheus/Grafana monitoring.

### 9. Retraining

* [ ] Detect model degradation and data drift.
* [ ] Fetch updated credit-risk data.
* [ ] Re-run preprocessing and feature engineering.
* [ ] Retrain and evaluate the model.
* [ ] Compare the new model against the production model.
* [ ] Promote the new model only when production criteria are satisfied.

### 10. Audit & Governance

* [ ] Log `PROSPECTID`, model version, feature version, prediction, probability, and timestamp.
* [ ] Make every prediction traceable to its model and feature pipeline.
* [ ] Protect sensitive financial and personal information.
* [ ] Keep credentials and database configuration outside source code.
* [ ] Maintain reproducible and auditable ML decisions.

### Production Flow

```text
Database → Fetch Prospect Data → Data Validation → Feature Engineering → Production Features → ML Model → LLM Model → Approved_Flag + Probability
→ API / Decisioning System → Monitoring & Audit → Drift Detection → Retraining
```
