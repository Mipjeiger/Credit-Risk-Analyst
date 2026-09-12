💳 Credit Risk Production System

ML Engineering · MLOps · LLMOps

Production-oriented credit-risk decisioning platform that connects Machine Learning, MLflow, FastAPI, RAG/LLM, Docker, Kubernetes, monitoring, and CI/CD into one system.

1. 🎯 Production Goal

The goal of this project is to move the existing credit-risk ML and LLM/RAG work from an experimentation environment into a production-oriented ML platform.

The system will combine three engineering areas:

ML Engineering → build, package, evaluate, and serve credit-risk models.

MLOps → manage model lifecycle, deployment, monitoring, and automation.

LLMOps → manage RAG, LLM deployment, evaluation, observability, and integration with the ML system.

The main principle is that each part of the platform has a clear responsibility.

The ML layer produces the quantitative credit-risk prediction. The policy layer turns that prediction into a business decision such as approval, rejection, or manual review. The RAG and LLM layer provides policy-grounded explanation and assistance when additional reasoning or context is needed. FastAPI exposes these capabilities to the application, while monitoring provides visibility into how the system behaves in production.

The LLM is therefore not responsible for replacing the ML model or becoming the primary source of the numerical risk score.

2. 📊 Current ML Assets

The existing production candidate is:

credit_risk_production/
└── database/
    └── LLM/
        └── outputs_llm/
            └── model_artifacts/
                └── model_bundle.joblib


The model_bundle.joblib currently contains six trained classification models:

Gradient Boosting
XGBoost
Decision Tree
Random Forest
Logistic Regression
K-Nearest Neighbors


It also contains the preprocessing required by those models:

StandardScaler
Label Encoders
Feature Columns
Class Labels


Therefore, this artifact already represents an important part of the ML production pipeline.

3. 🧠 ML Model Responsibility

The ML layer is responsible for producing a consistent quantitative prediction from the customer's available credit information.

The process begins with validated customer features. Those features are transformed using the same preprocessing logic used during model development. The approved production model then generates a predicted class and class probabilities. A standardized risk probability or score can be derived from those outputs according to the project's defined scoring policy.

For example:

predicted_flag = 1
risk_score = 0.9073
risk_score_100 = 90.73


The ML model does not decide how the business should handle every case.

That responsibility belongs to the policy/decision layer.

4. 🧪 MLflow — Model Lifecycle

MLflow will become the central location for managing the ML model lifecycle.

Instead of treating:

model_bundle.joblib


as the final production source, we will use it as the source artifact and register the models into MLflow.

Conceptually:

model_bundle.joblib
        │
        ├── Logistic Regression
        ├── Random Forest
        ├── Gradient Boosting
        ├── XGBoost
        ├── KNN
        └── Decision Tree
                 │
                 ▼
              MLflow
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
     Metrics  Artifacts  Versions
                 │
                 ▼
           Model Registry


MLflow will track:

Model information

Model name
Model version
Training configuration
Feature information
Dataset information


Evaluation

Accuracy
Precision
Recall
F1 Score
ROC AUC


Artifacts

model_bundle.joblib
model_features.json
metadata.json
best_parameters.json
feature importance
evaluation reports


This gives the project reproducibility and makes model deployment easier.

5. 🏆 Production Model Selection

The current results show:

ModelAccuracyF1 ScoreROC AUC







Gradient Boosting

99.53%

99.53%

99.99%

XGBoost

99.52%

99.52%

99.99%

Decision Tree

99.46%

99.46%

99.91%

Random Forest

98.98%

98.97%

99.98%

Logistic Regression

96.81%

96.72%

98.85%

KNN

77.58%

75.58%

89.83%

Gradient Boosting can initially become the production candidate, while the other models remain registered for comparison and future evaluation.

Production selection should not depend only on accuracy.

We should also consider:

reliability

inference latency

model stability

probability quality

business cost

explainability

operational complexity

6. 🔄 MLflow → FastAPI

Once the production model is registered, FastAPI should load the approved model version from MLflow rather than depending directly on a local notebook artifact.

This creates a controlled relationship between model development and model serving. A model can be trained and evaluated without immediately affecting production. Only after a specific version passes the required validation and approval checks should FastAPI use it.

This also makes rollback easier. If a newer model performs poorly or creates an operational problem, the serving layer can return to a previously validated model version instead of rebuilding the application around an old local file.

7. 🚀 FastAPI

FastAPI becomes the main application interface for the production system.

Location:

credit_risk_production/
└── app/
    └── api/


The API will connect the different parts of the system.

A request first needs to be validated so that the model receives the expected features and data types. The API then obtains the approved ML model and produces the risk prediction. The decision engine applies the relevant business policy to that prediction. When explanation or policy interpretation is required, the API can invoke the RAG and LLM layer. The resulting prediction, decision, model reference, and relevant LLM interaction information can then be persisted for auditability before the response is returned.

The API is therefore more than a simple prediction endpoint. It is the controlled application boundary through which the production decisioning workflow is executed.

Example API capabilities:

/health
/ready

/api/v1/predict
/api/v1/risk/score
/api/v1/risk/decision
/api/v1/llm/explain


The API therefore acts as the bridge between the ML system, decision engine, LLM system, and application.

8. 📋 Decision Engine

The production system should separate prediction from decision.

                 ML Model
                    ↓
            Risk Probability
                    ↓
             Policy Engine
                    ↓
       ┌────────────┼────────────┐
       ↓            ↓            ↓
 AUTO_APPROVE  MANUAL_REVIEW  AUTO_REJECT


For example:

Risk Score = 90.73
       ↓
Policy
       ↓
AUTO_APPROVE


Another customer might produce:

Risk Score = 45.20
       ↓
Policy
       ↓
MANUAL_REVIEW
       ↓
RAG + LLM


This makes the decision process easier to understand and audit.

9. 🤖 LLM Responsibility

The LLM is an intelligence and explanation layer, not the primary credit-risk model.

It should receive the relevant ML prediction, important customer risk factors, and information retrieved from the approved risk knowledge base. The RAG layer provides the policy context, while the LLM turns that information into a clear explanation or supporting recommendation.

For example, a manual-review case could be explained in terms of recent delinquency, utilization, or other relevant risk factors, while the explanation cites the applicable policy knowledge. This is more reliable than asking the LLM to make an independent credit decision from general language knowledge.

The LLM should not independently invent or replace the primary numerical credit-risk score. The ML prediction remains the authoritative quantitative signal, while the policy layer remains responsible for the formal business decision.

10. 📚 Existing RAG System

The current RAG assets are already available:

credit_risk_production/
└── database/
    └── LLM/
        ├── chroma_store/
        │
        └── outputs_llm/
            ├── llm_decisions.csv
            ├── model_artifacts/
            ├── pipeline_manifest.json
            └── rag_config.json


The Chroma store contains the vectorized knowledge used for retrieval.

The knowledge base can contain:

Delinquency Classification
Fraud Typologies & Red Flags
Regulatory Risk Policy Core
Scorecard Cut-off Policy


The purpose of RAG is to give the LLM access to the company's risk knowledge instead of relying only on its general language knowledge.

11. 🧠 LLMOps

LLMOps will manage the operational lifecycle of the LLM system.

The purpose is not simply to call an LLM from the application. The system should be able to explain which model was used, which RAG configuration and prompt were active, how the response performed, and whether the service experienced latency, errors, or fallback behavior.

The lifecycle therefore covers RAG configuration, LLM selection, evaluation, deployment, and production monitoring.

Important information to track includes:

LLM provider
LLM model
RAG version
Prompt version
Embedding model
Response latency
Token usage
Errors
Fallback events
Retrieved policy references


This allows us to understand not only whether the LLM works, but also how it behaves in production.

12. 🤗 Hugging Face LLM Deployment

The LLM will be deployed through Hugging Face as the model-serving layer.

FastAPI should communicate with the LLM service through a controlled interface rather than embedding model-specific logic throughout the application. The response should use a predictable structure so that the application can distinguish the explanation, supporting reasons, recommended action, and policy references.

The numerical ML risk score should remain separate from the generated language response. This prevents an LLM response from accidentally becoming the authoritative source for the underlying risk calculation.

Example:

{
  "risk_level": "MEDIUM_HIGH",
  "reason": [
    "Recent delinquency activity",
    "High credit utilization"
  ],
  "recommended_action": "MANUAL_REVIEW",
  "policy_references": [
    "Delinquency Classification"
  ]
}


The ML score remains the authoritative numerical prediction.

13. 🗄️ PostgreSQL 15 Alpine

PostgreSQL will store application-level production information.

Its purpose is to preserve what happened during real application usage. This can include prediction records, decision outcomes, the model version involved, LLM interaction metadata, and audit information.

MLflow and PostgreSQL therefore have different responsibilities. MLflow manages the lifecycle of ML experiments, artifacts, and registered model versions. PostgreSQL manages the business and application records created when the platform is actually used. Keeping these responsibilities separate makes the system easier to maintain and audit.

SystemResponsibility



MLflow

ML experiments, artifacts and model lifecycle

PostgreSQL

Application data, decisions and audit records

14. 🐳 Docker

Docker will package the production components into reproducible environments.

The main services can include FastAPI, MLflow, PostgreSQL, Prometheus, and Grafana, together with the other services required by the final deployment.

The main benefit is consistency. The application should behave according to a defined environment rather than depending on packages, Python versions, or configuration that happen to exist on a developer's machine. The same containerized application structure can then be tested before it is deployed to Kubernetes.

The important principle is:

The production application should not depend on a developer's local Python environment.

15. ☸️ Kubernetes

Kubernetes will run the production services.

Its role is to manage the running application rather than to replace the ML lifecycle tools. FastAPI, MLflow, PostgreSQL, Prometheus, Grafana, and other required services can be deployed as managed workloads.

Kubernetes provides service management, restart handling, health checks, scaling, deployment updates, and service discovery. This allows the ML application to behave like a production service instead of a collection of processes that must be started manually.

service management

restart handling

health checks

scaling

deployment updates

service discovery

FastAPI therefore becomes a deployable production service rather than a local development application.

16. 📈 Prometheus & Grafana

Production monitoring will cover both the technical health of the platform and the behavior of the ML and LLM components.

Technical monitoring answers whether the service is available and responsive. ML monitoring helps identify changes in prediction behavior and model usage. Decision monitoring shows how frequently the system approves, rejects, or sends cases for manual review. LLM monitoring shows whether the explanation service is healthy, responsive, and being used as expected.

API monitoring

Request count
Error rate
Latency
HTTP status


ML monitoring

Prediction volume
Model version
Inference latency
Prediction distribution


Decision monitoring

AUTO_APPROVE
MANUAL_REVIEW
AUTO_REJECT


LLM monitoring

LLM requests
LLM latency
LLM errors
Token usage
Fallback count


Prometheus collects the metrics.

Grafana visualizes them.

Production Services
       ↓
  Prometheus
       ↓
    Grafana
       ↓
Engineering Dashboard


17. 🔁 GitHub Actions

GitHub Actions will automate the software and model lifecycle so that changes are validated before they reach production.

The existing workflows have distinct responsibilities. CI focuses on code quality and testing. Model-serving automation focuses on validating model changes and connecting approved versions to the serving process. CD focuses on building the application image and deploying the validated application to the target environment.

Existing structure:

.github/
├── ci.yaml
├── cd.yaml
└── models_serving.yaml


CI

ci.yaml

Responsible for validating the project before deployment.

Code
 ↓
Tests
 ↓
Validation
 ↓
Docker Build


Model Serving

models_serving.yaml

Responsible for validating and deploying model changes.

Model Change
 ↓
Model Validation
 ↓
MLflow
 ↓
Production Model
 ↓
Serving


CD

cd.yaml

Responsible for application deployment.

GitHub
 ↓
Build
 ↓
Test
 ↓
Docker Image
 ↓
Kubernetes
 ↓
Production


18. 🔐 Production Configuration

Sensitive configuration should be separated from application code.

Examples include:

HUGGINGFACE_API_KEY
MLFLOW_TRACKING_URI
DATABASE_URL
POSTGRES_PASSWORD


These should be managed through environment variables and deployment secrets rather than committed into Git.

19. 🔍 Production Validation

Before a model or service becomes production-ready, it should pass several checks.

For the ML model, validation should confirm that the input schema is correct, predictions are valid, evaluation metrics meet the required standard, and the model can be served successfully through the API. Deployment should also be checked before the model is exposed to production traffic.

For the LLM, validation should confirm that retrieval returns relevant policy information, the generated response follows the expected structure, and the service meets acceptable latency and error requirements.

The goal is to create a controlled promotion process. A new model or LLM configuration should not directly affect production simply because it was successfully trained or generated a response.

20. 🔄 Complete Production Flow

The complete system becomes:

                 Customer Data
                       │
                       ▼
                 FastAPI Request
                       │
                       ▼
                 Feature Processing
                       │
                       ▼
                  MLflow Model
                       │
                       ▼
              Prediction + Probability
                       │
                       ▼
                 Policy Engine
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
           APPROVE   REVIEW   REJECT
                       │
                       ▼
                  RAG Retrieval
                       │
                       ▼
                 Hugging Face LLM
                       │
                       ▼
              Explanation / Action
                       │
                       ▼
                  PostgreSQL
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Prometheus            Audit Data
             │
             ▼
          Grafana


21. 🧩 Tool Responsibilities

ToolRole



Python / Scikit-learn / XGBoost

ML development

MLflow

ML tracking and model registry

Kubeflow

ML pipeline orchestration

FastAPI

Production API

PostgreSQL 15 Alpine

Application and audit database

ChromaDB

RAG vector store

Hugging Face

LLM deployment/inference

Docker

Containerization

Kubernetes

Production orchestration

Prometheus

Metrics collection

Grafana

Monitoring and visualization

GitHub Actions

CI/CD and model-serving automation

22. 👨‍💻 ML Engineer Scope

The ML Engineer work is primarily concerned with turning data and trained models into reliable inference assets.

This includes feature engineering, model training, evaluation, packaging, and making the approved model available for inference. In this project, the existing six-model bundle provides the starting point for establishing that production lifecycle.

The ML Engineer should also be responsible for validating whether the unusually high current evaluation results are trustworthy before those results are used as evidence for production selection.

23. ⚙️ MLOps Engineer Scope

The MLOps work is focused on making the ML system operationally reliable.

MLflow manages the model lifecycle and registry, while Kubeflow can orchestrate repeatable ML workflows. Docker provides consistent environments, Kubernetes manages deployed services, GitHub Actions automates validation and deployment workflows, and Prometheus/Grafana provide operational visibility.

The objective is to make the ML system:

reproducible

deployable

observable

versioned

maintainable

24. 🧠 LLMOps Engineer Scope

The LLMOps work is:

RAG
 ↓
ChromaDB
 ↓
LLM
 ↓
Hugging Face
 ↓
Evaluation
 ↓
Monitoring


The objective is to make the LLM system:

versioned

observable

reproducible

policy-grounded

safely integrated with the ML decision system

25. 🚀 Production Development Order

The production implementation should follow this order:

Phase 1 — ML Production

Validate model_bundle.joblib

Create production model service

Register the six models in MLflow

Track metrics and artifacts

Select production candidate

Validate production model

Phase 2 — API

Build FastAPI application

Add prediction endpoint

Add risk decision endpoint

Connect API to MLflow

Add PostgreSQL persistence

Phase 3 — Decision Engine

Implement policy rules

Implement AUTO_APPROVE

Implement AUTO_REJECT

Implement MANUAL_REVIEW

Phase 4 — LLMOps

Connect existing ChromaDB

Connect RAG retrieval

Connect FastAPI to Hugging Face

Add structured LLM responses

Add LLM evaluation

Track LLM interaction metadata

Phase 5 — Infrastructure

Dockerize services

Configure PostgreSQL 15 Alpine

Deploy services to Kubernetes

Configure service health checks

Phase 6 — Observability

Add Prometheus metrics

Build Grafana dashboards

Monitor ML inference

Monitor decision routes

Monitor LLM latency and errors

Phase 7 — CI/CD

Implement ci.yaml

Implement models_serving.yaml

Implement cd.yaml

Automate model validation

Automate Docker builds

Automate Kubernetes deployment

26. 🎯 Final Target

The final result is not simply a credit-risk model.

It becomes a production-oriented AI risk decisioning platform:

                    CREDIT RISK PLATFORM

       ┌─────────────────────────────────────┐
       │          ML ENGINEERING             │
       │                                     │
       │ Feature, model development, and evaluation       │
       └──────────────────┬──────────────────┘
                          │
                          ▼
                     ┌─────────┐
                     │ MLflow  │
                     └────┬────┘
                          │
                          ▼
                    Production ML
                          │
                          ▼
                     ┌─────────┐
                     │ FastAPI │
                     └────┬────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          Policy Engine          RAG + LLM
                │                   │
                │              Hugging Face
                │                   │
                └─────────┬─────────┘
                          ▼
                   Risk Decision
                          │
                          ▼
                     PostgreSQL
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
           Prometheus            Audit
                │
                ▼
             Grafana

       ─────────────────────────────────────

       MLOps:
       MLflow + Kubeflow + Docker + Kubernetes
       + GitHub Actions + Prometheus + Grafana

       LLMOps:
       RAG + ChromaDB + Hugging Face
       + Evaluation + Monitoring


Production principle: ML produces the quantitative risk signal, the policy engine controls the business decision, and the LLM provides policy-grounded explanation and assistance.