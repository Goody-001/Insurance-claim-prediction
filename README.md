# Building Insurance Claim Prediction

## Project Overview

This project focuses on building a predictive model to determine whether a building is likely to have an insurance claim during a given insured period. The objective is to predict the probability of a building having at least one claim based on its characteristics. The target variable, **Claim**, is defined as:

* `1` if the building has at least one claim during the insured period
* `0` if the building does not have any claim

The predictions from this model can help insurance companies assess risk, optimize premiums, and improve decision-making for underwriting policies.

---

## Dataset

The dataset includes building-level features (categorical and numerical) that describe each building. The features are used to predict the likelihood of an insurance claim occurring.

---

## Project Steps

### 1. Data Cleaning and Preprocessing

* Handled missing values
* Encoded categorical variables using `OneHotEncoder`
* Standardized and normalized numerical variables where necessary
* Split the data into training and testing sets

### 2. Exploratory Data Analysis (EDA)

* Analyzed the distribution of numerical features using histograms and boxplots
* Explored relationships between categorical features and the target variable using countplots and bar charts

### 3. Feature Engineering

* Selected relevant features based on EDA insights
* Created interaction terms and transformations where applicable
* Applied preprocessing pipelines using `scikit-learn`'s `ColumnTransformer` for streamlined model input

### 4. Model Implementation

The following models were implemented and experimented with:

1. **Logistic Regression** – a baseline probabilistic model for binary classification
2. **Random Forest Classifier** – an ensemble method to improve predictive performance and reduce overfitting
3. **Gradient Boosting Classifier** – a boosting algorithm to optimize predictions through sequential learning

Hyperparameter tuning was performed for all models using techniques such as GridSearchCV and RandomizedSearchCV to identify the best parameters.

### 5. Model Evaluation

Models were evaluated using:

* ROC-AUC Score for probabilistic predictions
* Classification reports for all models to summarize precision, recall, F1-score, and support


---

## Libraries Used

* **Pandas** – for data manipulation and analysis
* **NumPy** – for numerical computations
* **Matplotlib** and **Seaborn** – for data visualization
* **Scikit-learn** – for machine learning models, preprocessing, and evaluation metrics

---

## How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook `Building_Insurance_Claim_Prediction.ipynb` to reproduce the analysis and model results.

---

## Outcome

The project demonstrates the end-to-end process of building a predictive model, from data preprocessing and exploratory analysis to model training, hyperparameter tuning, and evaluation. The insights and model can be used by insurance companies to assess risk and inform decisions.

---

## Author

**Goodluck Nwachukwu**
Mechanical Engineer & Data Analyst
