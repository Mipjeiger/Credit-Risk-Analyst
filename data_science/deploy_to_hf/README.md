---
title: Credit Risk Agentic AI
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# 🏦 Credit Risk Agentic AI

An **Agentic AI system for credit-risk analysis** that combines traditional machine learning, explainable AI, Retrieval-Augmented Generation (RAG), and an instruction-tuned Large Language Model (LLM).

The system is designed to demonstrate how an LLM can work together with an existing credit-risk ML model rather than replacing it.

The core principle is:

> **The ML model makes the prediction. SHAP explains the prediction. RAG provides policy evidence. The LLM acts as the reasoning layer that combines these outputs into a grounded explanation.**

---

## 🎯 Objective

The objective of this project is to build and deploy an **Agentic AI system for credit-risk analysis** on Hugging Face.

Instead of building a generic chatbot, the LLM acts as an agent that can use specialized tools.

For example, when a user asks:

> **"Why was this applicant rejected?"**

the agent can:

1. Run the credit-risk ML model.
2. Obtain the prediction and probability.
3. Generate SHAP feature contributions.
4. Retrieve relevant credit-risk policies through RAG.
5. Provide the evidence to the LLM.
6. Generate a natural-language explanation.
7. Recommend an appropriate next action based on the available evidence.

---

# 🧠 System Architecture

```text
                    User
                      │
                      ▼
              ┌───────────────┐
              │   Gradio UI   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Agent / LLM   │
              │ Reasoning     │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │   ML    │   │  SHAP   │   │   RAG   │
   │  Model  │   │ Explain │   │ Policy  │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        ▼             ▼             ▼
   Prediction    Feature Impact   Policy
                                  Evidence
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
              ┌───────────────┐
              │      LLM      │
              │ Explanation   │
              └───────┬───────┘
                      │
                      ▼
                Final Answer

🤖 Machine Learning LayerThe ML model is responsible for the credit-risk prediction.The target variable is:PlaintextApproved_Flag
The project experiments with multiple classical machine-learning algorithms:XGBoostRandom ForestGradient BoostingLogistic RegressionDecision TreeK-Nearest NeighborsThe model produces structured outputs such as:JSON{
  "approved_flag": 0,
  "risk_probability": 0.87
}
The LLM does not replace this prediction.🔍 Explainable AI with SHAPSHAP is used to understand the contribution of individual features to the ML prediction.For example:PlaintextTop contributing features:

Tot_Missed_Pmnt       +0.31
Tot_Active_TL         +0.18
pct_active_tl         +0.12
Total_TL_opened_L6M   -0.08
These structured results are passed to the agent.The LLM can then explain the prediction using the actual model evidence instead of inventing feature importance.WorkflowPlaintextApplicant Features
       │
       ▼
Credit Risk Model
       │
       ▼
Prediction
       │
       ▼
SHAP Explainer
       │
       ▼
Feature Contributions
       │
       ▼
LLM Explanation
📚 Policy RAGThe system uses Retrieval-Augmented Generation (RAG) to provide domain-specific credit-risk knowledge to the LLM.The knowledge base can contain documents such as:PlaintextPolicy Knowledge Base
│
├── Delinquency Classification
├── Fraud Typologies & Red Flags
├── Regulatory Risk Policy Core
└── Scorecard Cut-off Policy
Instead of relying only on the LLM's pretrained knowledge, the system retrieves relevant policy information before generating an answer.RAG workflowPlaintextUser Question
      │
      ▼
Retrieve Relevant Documents
      │
      ▼
Relevant Policy Context
      │
      ▼
Agent / LLM
      │
      ▼
Grounded Explanation
🧠 Agentic AIThe key difference between this project and a conventional chatbot is tool usage.The LLM can orchestrate specialized tools.Credit Risk ToolPythonpredict_credit_risk(features)
Responsible for obtaining the ML prediction.SHAP ToolPythonexplain_prediction(features)
Responsible for generating feature contributions.Policy Retrieval ToolPythonretrieve_policy(query)
Responsible for retrieving relevant policy information.The conceptual workflow is:Plaintext                  Agent
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   ML Model       SHAP          RAG
     Tool         Tool          Tool
       │            │            │
       ▼            ▼            ▼
 Prediction     Explainability  Evidence
       │            │            │
       └────────────┼────────────┘
                    ▼
                   LLM
                    │
                    ▼
             Final Explanation
💬 Example Use CaseUserPlaintextWhy was applicant #123 rejected?
Agent workflowPlaintext1. Identify the applicant
2. Retrieve applicant features
3. Run the ML model
4. Obtain Approved_Flag
5. Generate SHAP contributions
6. Search relevant risk policies
7. Combine ML + SHAP + RAG evidence
8. Ask the LLM to generate an explanation
Example responsePlaintextThe applicant was classified as not approved by the
credit-risk model.

The main contributing factors were elevated missed
payments and a high proportion of active credit accounts.

The retrieved policy information indicates that these
factors may require additional review depending on the
applicable policy thresholds.

Recommended action:
Review the applicant's credit history and supporting
documentation.
The important distinction is that the LLM explains the existing model result rather than independently creating the credit-risk decision.🗃️ DatasetThe project uses merchant/application-level credit-risk data.Example features include:PlaintextTotal_TL
Tot_Closed_TL
Tot_Active_TL
Total_TL_opened_L6M
Tot_TL_closed_L6M
pct_tl_open_L6M
pct_tl_closed_L6M
pct_active_tl
pct_closed_tl
Tot_Missed_Pmnt
Auto_TL
CC_TL
Target:PlaintextApproved_Flag
The dataset contains financial and credit-history information that is transformed into features for the machine-learning models.🧩 Technology StackThis project intentionally focuses on the Data Science, Machine Learning, Explainable AI, RAG, and LLM layers.LayerTechnologyData ProcessingPython, Pandas, NumPyMachine LearningScikit-learn, XGBoostExplainable AISHAPLLMInstruction-tuned LLMAgent FrameworkLangGraphRAGChromaDB / Vector StoreEmbeddingsSentence Transformers / Embedding ModelLLM InferenceHugging FaceApplication UIGradioDeploymentHugging Face Spaces🚀 Hugging Face DeploymentThe application is deployed using Hugging Face Spaces with Gradio.The Space contains the application and agent logic, while the LLM is accessed through Hugging Face inference infrastructure.Plaintext             Hugging Face Space
                    │
                    ▼
              ┌───────────┐
              │  Gradio   │
              │    UI     │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │   Agent   │
              └─────┬─────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       ML          SHAP        RAG
      Model        Tool       Search
        │           │           │
        └───────────┼───────────┘
                    ▼
                  LLM
                    │
                    ▼
              Final Response
📦 Application StructurePlaintext.
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   └── xgboost.joblib
│
├── tools/
│   ├── credit_risk.py
│   ├── shap_explainer.py
│   └── policy_retriever.py
│
├── rag/
│   ├── documents/
│   └── vector_store/
│
└── config/
    └── settings.py
🔄 End-to-End Data Science + LLM WorkflowPlaintext                    Raw Data
                       │
                       ▼
                Data Processing
                       │
                       ▼
                 Feature Data
                       │
                       ▼
             Machine Learning Model
                       │
                       ▼
              Credit Risk Prediction
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
          SHAP                Policy RAG
       Explanation            Retrieval
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                  Agent / LLM
                       │
                       ▼
             Grounded Explanation
                       │
                       ▼
                  Gradio UI
🎯 Design PrincipleThe system separates the responsibilities of the ML model and the LLM.Machine LearningPlaintextML Model
   │
   └──► Predict credit-risk outcome
Explainable AIPlaintextSHAP
 │
 └──► Explain model prediction
RAGPlaintextPolicy Documents
 │
 └──► Provide domain evidence
LLMPlaintextLLM
 │
 ├──► Understand user question
 ├──► Orchestrate tools
 ├──► Interpret ML results
 ├──► Interpret SHAP results
 ├──► Use retrieved policy evidence
 └──► Generate natural-language explanation
This separation helps prevent the LLM from becoming an unsupported source of credit-risk decisions.⚠️ LimitationsThis project is a Data Science and AI engineering demonstration and should not be considered a production credit-decisioning system.Important considerations for real-world deployment include:Data qualityModel bias and fairnessModel driftPolicy changesExplainability limitationsRAG retrieval qualityLLM hallucinationPrompt injectionSensitive financial informationRegulatory requirementsHuman review and governanceThe LLM should not be treated as an independent authority for credit approval decisions.📌 Project SummaryThis project demonstrates the integration of:PlaintextData Science
     +
Machine Learning
     +
Explainable AI
     +
RAG
     +
Agentic AI
     +
LLM
     +
Hugging Face Deployment