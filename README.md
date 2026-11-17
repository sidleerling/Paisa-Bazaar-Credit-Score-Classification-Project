# Paisa-Bazaar-Credit-Score-Classification-Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange.svg)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-App-brightgreen.svg)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/Sidlyer/Paisabazaar%20Credit%20Score%20Predictor)

An end-to-end Machine Learning project developed for **Paisabazaar**, one of India's leading financial marketplaces for loans and credit cards. This project focuses on assessing customer creditworthiness through predictive classification modeling and deploys an interactive web application for real-time credit score prediction and financial advice.

---

## 📋 Table of Contents
- [Business Context & Objective](#-business-context--objective)
- [Project Workflow](#-project-workflow)
- [Dataset Overview](#-dataset-overview)
- [Exploratory Data Analysis (EDA) & Insights](#-exploratory-data-analysis-eda--insights)
- [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Model Tuning](#-model-tuning)
- [App Deployment & Live Demo](#-app-deployment--live-demo)
- [Repository Structure](#-repository-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [Author](#-author)

---

## 🏢 Business Context & Objective

Paisabazaar assists customers in discovering, comparing, and applying for diverse banking and credit products. A critical part of their operations involves assessing customer **creditworthiness** for loan approvals and risk management. 

### What is a Credit Score?
A credit score is a numerical representation of an individual's creditworthiness, indicating their capability and probability of returning borrowed money with interest on time[cite: 3]. Scores typically range from 300 to 850 (categorized in this dataset into **Poor**, **Standard**, and **Good**)[cite: 5, 6]. 
* **Good Scores:** Indicate low risk[cite: 5], resulting in easier loan approvals, lower interest rates, and higher credit limits[cite: 2].
* **Poor Scores:** Indicate high risk[cite: 5] and greater likelihood of defaulting[cite: 5], which can lead to higher interest rates[cite: 5] or application rejections.

### Project Goals
1. **Predictive Modeling:** Train a robust classification model to categorize customers into **Poor**, **Standard**, or **Good** credit scores based on financial and credit profiles[cite: 6].
2. **Web Application Deployment:** Build and deploy an interactive web-based app using **Gradio**[cite: 6] hosted on **Hugging Face Spaces**[cite: 33] where users can input credit details to receive instant score predictions and confidence metrics[cite: 33].
3. **Value Addition:** Empower customers with insights to improve their financial health while helping Paisabazaar mitigate loan default risks and personalize financial offerings[cite: 6].

---

## 🔄 Project Workflow

The project follows a standard data science pipeline:
1. **Exploratory Data Analysis (EDA):** Univariate, bivariate, and multivariate analysis with visual representations[cite: 8].
2. **Data Preprocessing & Cleaning:** Handling datatypes, outliers, and unrealistic records[cite: 8, 9].
3. **Feature Engineering & Selection:** Creating impactful domain-specific features and reducing multicollinearity[cite: 8, 18, 19, 20].
4. **Class Imbalance Treatment:** Applying **SMOTE** to balance underrepresented classes[cite: 10, 24].
5. **Model Training & Evaluation:** Benchmarking Logistic Regression, Decision Tree, and Random Forest classifiers[cite: 25, 26, 27, 28].
6. **Model Tuning:** Feature selection with `SelectKBest` and hyperparameter optimization using `RandomizedSearchCV`[cite: 30].
7. **Deployment:** Building and hosting an interactive Gradio application[cite: 33].

---

## 📊 Dataset Overview

- **Source:** Obtained via Labmentix[cite: 7].
- **Size:** 100,000 records across 27 features[cite: 7].
- **Key Feature Categories:**
  - **Personal Information:** `Customer_ID`, `Month`, `Name`, `Age`, `SSN`, `Occupation`[cite: 7]
  - **Financial Profile:** `Annual_Income`, `Monthly_Inhand_Salary`, `Amount_Invested_Monthly`, `Monthly_Balance`[cite: 7]
  - **Credit Profile:** `Num_of_Bank_Accounts`, `Num_Credit_Cards`, `Num_of_Loan`, `Type_of_Loan`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`, `Outstanding_Debt`, `Credit_Utilization_Ratio`, `Total_EMI_per_Month`[cite: 7]
  - **Payment Behavior & History:** `Interest_Rate`, `Delay_from_due_date`, `Num_of_Delayed_Payments`, `Credit_History_Age`, `Payment_of_Min_Amount`, `Payment_Behaviour`[cite: 7]
  - **Target Variable:** `Credit_Score` (`Poor`, `Standard`, `Good`)[cite: 7]

---

## 🔍 Exploratory Data Analysis (EDA) & Insights

Key patterns and business insights derived from data visualizations:
- **Annual Income vs. Credit Score:** Customers with a *Good* credit score earn significantly higher (approx. 58.22% more) than those with a *Poor* score, reflecting better debt diversification and repayment capacity[cite: 14].
- **Age Distribution:** The vast majority of customers fall between 18–45 years old, representing Paisabazaar's prime financially active segment[cite: 12, 14].
- **Interest Rates:** Customers with *Poor* credit scores experience up to 200% higher median interest rates compared to those with *Good* scores as lenders compensate for risk[cite: 14].
- **Credit History Age:** Individuals with a *Good* credit score have roughly 70% longer median credit history ages, demonstrating extended financial reliability[cite: 15].
- **EMI to Income Ratio:** Engineered to measure debt burden; high ratios correlate heavily with financial stress and *Poor* credit scores[cite: 15].

---

## 🛠️ Data Preprocessing & Feature Engineering

- **Data Type Corrections:** Converted float features representing countable entities (e.g., age, number of bank accounts, active loans, delayed payments) into proper integers for semantic accuracy[cite: 9, 22].
- **Unrealistic Data Removal:** Filtered out logically inconsistent records, such as customers under 18 years old holding professional titles like developers or lawyers[cite: 9].
- **Feature Engineering:**
  - `EMI_to_Income_Ratio`: Calculated by dividing `Total_EMI_per_month` by `Monthly_Inhand_Salary` to gauge debt strain[cite: 18].
  - `Loan_Types`: Processed and counted distinct loan types from string categories to capture credit variety[cite: 19].
- **Feature Reduction:** Dropped redundant or highly collinear columns (e.g., `Monthly_Inhand_Salary` vs `Annual_Income`, and `Amount_Invested_Monthly`) along with uninformative personal identifiers (`Customer_ID`, `Name`, `SSN`)[cite: 20].
- **Class Imbalance Handling:** Addressed heavy class skew (where *Standard* made up the overwhelming majority) using **SMOTE (Synthetic Minority Over-sampling Technique)** to enrich minority classes (*Poor* and *Good*) without mere duplication[cite: 10, 24].

---

## 🏆 Model Training & Evaluation

Three classification models were trained and evaluated on metrics including **Accuracy, Precision, Recall, and F1-Score**[cite: 25]:

| Model | Accuracy Score | Key Observations |
| :--- | :---: | :--- |
| **Logistic Regression** | 64.8% | Served as a fast, interpretable baseline model[cite: 26]. |
| **Decision Tree Classifier** | 71.1% | Handled non-linear interactions well and proved robust to outliers[cite: 27]. |
| **Random Forest Classifier** | **80.8%** | **Best performing model**, significantly outperforming others across all classes by capturing complex feature interactions[cite: 28, 29]. |

---

## ⚙️ Model Tuning

The **Random Forest Classifier** was selected for further refinement[cite: 30, 32]:
- **Feature Selection:** Applied `SelectKBest` (using Chi-Squared test) alongside domain knowledge to distill down to 14 optimal features[cite: 30].
- **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` to optimize parameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`[cite: 30].
- **Tuned Model Performance:** Achieved an improved accuracy of **81.5%** with better precision/recall balance and lower false-negative rates for minority classes[cite: 31].

---

## 🚀 App Deployment & Live Demo

The optimized Random Forest model was serialized using `Joblib` and integrated into an interactive web application built with **Gradio**. 

### App Features:
- Clean UI sliders and dropdown fields for entering financial parameters.
- Predicts final credit score category (*Poor*, *Standard*, or *Good*)[cite: 33, 35].
- Displays a **Prediction Confidence Breakdown** to communicate model uncertainty transparently[cite: 33, 35].

🔗 **Live Application Links:**
- [Hugging Face Space - Paisabazaar Credit Score Predictor](https://huggingface.co/spaces/Sidlyer/Paisabazaar%20Credit%20Score%20Predictor)[cite: 36]

---

## 📁 Repository Structure

```text
├── dataset/                    # Dataset files (if applicable)
├── models/                     # Saved model checkpoints (.pkl / joblib)
├── app.py                      # Gradio web application script
├── requirements.txt            # Required python dependencies
├── Credit_Score_Prediction.ipynb # Jupyter Notebook containing complete EDA & modeling code
└── README.md                   # Project documentation
