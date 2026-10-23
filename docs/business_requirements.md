# FinSight AI — Business Requirements Document

**Version:** 0.1
**Status:** Initial draft
**Project type:** End-to-end Data Science and AI prototype

## 1. Business background

A digital lending business wants to improve its understanding of credit risk and loan application journeys. Risk analysts need useful estimates of payment-difficulty risk, while business managers need visibility into application conversion and drop-off.

## 2. Business objectives

1. Estimate the probability of the dataset-defined payment-difficulty outcome for eligible applications.
2. Explain important factors influencing model predictions.
3. Analyze application-stage conversion and drop-off using suitable event data.
4. Provide an accessible interface for predictions and business insights.
5. Demonstrate a reproducible, testable, deployable ML and AI workflow.

## 3. Intended users

* Risk analysts
* Business managers
* Data Scientists
* ML / platform engineers
* Governance and compliance reviewers

## 4. Scope

**In scope**

* Data acquisition, validation, and exploratory analysis
* Feature engineering and leakage prevention
* Baseline and candidate classification models
* Model evaluation, calibration assessment, and explainability
* Prediction API and user interface
* Synthetic loan-journey analytics
* A document-grounded AI assistant
* Automated tests, containerization, deployment, and basic monitoring

**Out of scope**

* Production credit approvals or loan rejections
* Real customer data or confidential employer data
* Claims of verified financial savings
* A production-grade credit decision engine
* Guaranteed prediction accuracy or guaranteed business impact

## 5. ML problem definition

**Task:** Binary classification.

**Target:** The Home Credit dataset's documented `TARGET` outcome.

**Prediction output:** Estimated probability of the positive class, plus a documented risk interpretation.

**Prediction point:** Application scoring using only eligible information available at that point.

## 6. Success measures

* Establish a DummyClassifier baseline.
* Evaluate candidate models using ROC-AUC, average precision, precision, recall, and appropriate additional diagnostics.
* Select a decision threshold based on a documented use case and trade-offs.
* Assess probability calibration where relevant.
* Validate API behavior with automated tests.
* Demonstrate the application running in a separate browser or environment.
* Document limitations, reproducibility, and monitoring.

No model performance or business-impact result will be claimed until it has been measured.

## 7. Key assumptions and risks

* The public dataset is a learning proxy, not necessarily representative of a current lending portfolio.
* Feature availability and target definitions must be verified.
* Class imbalance, data leakage, bias, and distribution shift may affect results.
* Credit-related predictions require careful governance and human oversight.
* Hosting availability and resource constraints will be checked before deployment.

## 8. Deliverables

1. Reproducible data analysis and preparation workflow
2. Trained and evaluated model artifact
3. Prediction API and Streamlit interface
4. Customer-journey analytics dashboard
5. Document-grounded AI assistant
6. Tests, Docker configuration, deployment instructions, and monitoring
7. Technical documentation and interview walkthrough

## 9. Definition of done

The project is complete when the core application can be reproduced, predictions can be requested and explained, tests pass, deployment is demonstrable, and limitations and evaluation results are documented honestly.
