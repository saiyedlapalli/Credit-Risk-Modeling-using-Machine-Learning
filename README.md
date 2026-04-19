# Credit Risk Modeling Using Machine Learning

This project focuses on predicting whether a borrower is likely to fully repay a loan or end up being charged off, using historical LendingClub loan data and machine learning techniques.

The goal was not just to train one model and stop there, but to walk through the full problem like a real credit-risk exercise: understand the data, clean it carefully, engineer usable features, build a neural network model, and compare it against a few traditional machine learning algorithms.


## Project objective

Build a classification model that can learn patterns from past LendingClub loans and predict loan repayment behavior.

More specifically, this project aims to:
- clean and prepare raw lending data for modeling
- explore which borrower and loan features appear related to repayment outcomes
- transform categorical and numerical variables into a model-ready format
- train a deep learning model for binary classification
- compare its performance with several standard machine learning models


## Dataset

The notebook uses the **LendingClub loan dataset** along with a metadata file that explains the variables.

Files referenced in the notebook:
- `lending_club_loan_two.csv`
- `lending_club_info.csv`

The target variable is:
- `loan_status`

In the notebook, `loan_status` is converted into a binary variable:
- `1` = Fully Paid
- `0` = not Fully Paid / charged off class


## What I did in this project

### 1. Exploratory Data Analysis
I started by understanding the structure of the data and how the target variable relates to the input features.

Some of the analysis included:
- distribution of loan amounts
- correlations across numerical features
- boxplots of `loan_amnt` vs `loan_status`
- countplots of `grade` vs `loan_status`
- examining how charge-off behavior changes across credit grades

One clear pattern from the analysis was that as loan grade worsens, the share of bad outcomes increases.


### 2. Data cleaning and preprocessing
A large part of this project was making the raw data usable for machine learning.

Key preprocessing steps included:
- converting `loan_status` into a binary target
- converting `term` into an integer
- removing columns that were redundant or not practical for modeling, such as:
  - `sub_grade`
  - `emp_title`
  - `title`
- cleaning `emp_length` and then dropping it after checking that it did not show a strong relationship with the target
- filling missing values in `mort_acc` using the mean
- dropping the small remaining share of missing rows
- reducing category noise in `home_ownership` by grouping `NONE` and `ANY` into `OTHER`
- extracting ZIP code from the address field and dropping the original `address`
- simplifying `earliest_cr_line` to just the year
- converting `pub_rec` into a simpler risk-style indicator


### 3. Encoding categorical features
After cleaning, the remaining object-type columns were encoded into numeric form using an ordinal mapping approach.

This made the data compatible with machine learning models while keeping the workflow simple and easy to reproduce inside the notebook.


### 4. Feature scaling and train-test split
Before modeling:
- features were separated into `X` and `Y`
- the data was split into training and testing sets
- a **MinMaxScaler** was applied so the model could learn from normalized inputs


### 5. Deep learning model
The main model in this project is a **feedforward neural network built with TensorFlow/Keras**.

Architecture used in the notebook:
- Dense layer: 78 units, ReLU
- Dropout: 0.2
- Dense layer: 39 units, ReLU
- Dropout: 0.2
- Dense layer: 19 units, ReLU
- Dropout: 0.2
- Dense layer: 19 units, ReLU
- Dropout: 0.2
- Output layer: 1 unit, Sigmoid

Training setup:
- loss function: `binary_crossentropy`
- optimizer: `adam`
- early stopping based on validation loss
- batch size: `256`
- epochs: up to `25`

The use of dropout and early stopping helps reduce overfitting and improve generalization.


### 6. Model evaluation
The notebook evaluates model performance using:
- classification report
- confusion matrix
- ROC-AUC score

These metrics help assess how well the model distinguishes between good and bad loans, rather than relying only on accuracy.


### 7. Benchmark model comparison
To make the project more useful, I also compared the neural network against a set of traditional classifiers:
- Linear SVC
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest Classifier

This is important because in real modeling work, a more complex model is only valuable if it performs better than simpler baselines.


## Tools and libraries used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras


## Notes and limitations

This project is a solid end-to-end classification workflow, but there are a few areas that could be improved further:

- ordinal encoding was used for categorical variables, but one-hot encoding or target encoding may work better for some features
- class imbalance should be studied more carefully and could be handled with class weights or resampling
- model tuning was limited, so there is room for hyperparameter optimization
- feature importance and interpretability could be added for stronger business value
- cross-validation would make the evaluation more robust


## Future improvements

Here are some strong next steps if this project is expanded:
- build a proper **probability of default (PD)** modeling pipeline
- use **one-hot encoding** for nominal categories
- tune the neural network architecture systematically
- evaluate precision-recall tradeoffs for lending decisions
- add **feature importance / SHAP analysis**
- create a simple dashboard for loan risk scoring
- compare against gradient boosting models like XGBoost or LightGBM


## What this project demonstrates

This project shows experience with:
- credit-risk style problem solving
- data cleaning and feature engineering
- exploratory analysis of lending data
- binary classification modeling
- neural networks for tabular data
- baseline model comparison
- model evaluation using business-relevant metrics

