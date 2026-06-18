# AGENTS.md

## Project Focus

This project uses iM Bank anonymized corporate monthly panel data for an ML/DL project:

**predicting online banking activation potential for corporate customers.**

Online banking is defined as:

* `인터넷뱅킹거래금액`
* `스마트뱅킹거래금액`

`자동이체금액` may be used as an optional supporting digital-activity feature, not as the main target.

---

## Target Definition

Main target:

**Next-month increase in online channel share**

Definitions:

* Online amount = `인터넷뱅킹거래금액` + `스마트뱅킹거래금액`
* Total channel amount = `인터넷뱅킹거래금액` + `스마트뱅킹거래금액` + `창구거래금액` + `ATM거래금액` + `폰뱅킹거래금액`
* Online share = online amount / total channel amount
* Target = 1 if next month online share - current month online share >= 0.05
* This means a 5 percentage-point increase.

If total channel amount is 0:

* Set online share to 0
* Add a flag such as `is_no_channel_activity`

Secondary target:

**Next-month new online banking activation**

* Current month online amount = 0
* Next month online amount > 0

---

## Data Setup Rules

Use a time-based panel setup:

* Row unit: one corporation in one month
* Feature point: month `t`
* Prediction point: month `t+1`
* Exclude the final month when creating next-month targets
* Split train/validation/test by time, not random split
* Never use future-month data in features
* Rolling features must use only current and past months up to month `t`

Recommended split:

* Train: 2023-01 to 2024-12
* Validation: 2025-01 to 2025-06
* Test: 2025-07 to 2025-11

---

## Core Features

Focus feature engineering on:

* Recent 3-month and 6-month online usage
* Online channel share and its change
* Branch channel share and its change
* Internet vs smart banking usage balance
* Transaction activity level
* Deposit and loan scale
* Industry, region, customer grade, and managed-customer flag
* Optional channel-behavior cluster labels

Use safe division for ratio features. If the denominator is 0, handle it explicitly.

---

## ML Strategy

Do not rely only on AutoML.

Recommended flow:

1. Train simple baseline models manually:

   * Logistic Regression
   * Decision Tree or RandomForest

2. Train at least one main tabular model manually:

   * LightGBM
   * XGBoost
   * CatBoost

3. Use AutoML only as a comparison framework:

   * FLAML
   * PyCaret
   * AutoGluon

Compare manual models and AutoML results using the same data split and metrics.

---

## DL Strategy

Compare:

* MLP baseline
* LSTM or GRU using recent monthly sequences
* Optional 1D-CNN or Transformer Encoder

DL input should use recent 6-month or 12-month customer transaction sequences.

Compare DL models with ML models to check whether monthly sequence information improves performance.

---

## Evaluation Metrics

Use:

* ROC-AUC
* PR-AUC
* F1-score
* Top-K Recall
* Lift@K

Prioritize:

* `Top-K Recall`
* `Lift@K`
* `PR-AUC`

Reason: the business use case is campaign target selection, so the model should identify high-potential customers in the top predicted group.

Accuracy should not be the main metric.

---

## Interpretation

Use:

* Feature importance
* SHAP
* Top-K customer profile analysis

Final results should explain which factors are related to online banking activation and how the model can support DX campaign targeting.

---

## Coding Guidelines

* Use notebooks for EDA and visualization.
* Move finalized preprocessing, target creation, feature engineering, and model training into Python files.
* Use fixed random seeds.
* Do not overwrite raw data.
* Save metrics and model comparison results.
* Keep Korean column names unless a mapping file is created.
