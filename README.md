<h2>Car Insurance Claim Prediction</h2>
This project aims to predict car insurance claims using machine learning models. The dataset, sourced from Kaggle, includes various features related to car insurance policies and claims.

<h3>Project Overview</h3>
<b>Data Preparation:</b>

The dataset is loaded and cleaned, including handling missing values and removing duplicates.
String columns are split and converted to numeric values.
Unnecessary columns are dropped.<br>

<b>Exploratory Data Analysis (EDA):</b>

Summary statistics and visualizations are used to understand the data distribution and identify patterns.
The dataset is categorized into symbolic, discrete, and continuous fields.<br>

<b>Data Transformation:</b>

One-hot encoding is applied to categorical variables.
Correlation analysis is performed to identify and remove redundant features.<br>

<b>Balancing the Dataset:</b>

Techniques such as oversampling and undersampling are used to handle class imbalance in the target variable.<br>

<b>Modeling:</b>

Decision Tree and Random Forest models are built and trained on the balanced datasets.
The models are fine-tuned and their performance is evaluated using metrics like accuracy, precision, recall, and F1 score.<br>

<b>Evaluation:</b>

Confusion matrices and evaluation metrics are calculated for each model.
Visualizations of the model performance and feature importance are created to compare results.
