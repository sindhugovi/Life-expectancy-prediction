# life-expectancy-prediction
Project Overview

This project aims to analyze and preprocess the "Life Expectancy Data" dataset and build regression models to predict life expectancy based on various features. The steps include exploratory data analysis (EDA), data preprocessing, and training multiple regression models to evaluate their performance.

Dataset

The dataset used in this project is "Life Expectancy Data.csv." It contains information related to life expectancy, including health, economic, and demographic factors.

Dependencies

The project uses the following Python libraries:

numpy

pandas

seaborn

matplotlib

scikit-learn

Install the required libraries using:

pip install numpy pandas seaborn matplotlib scikit-learn

Steps in the Project

1. Data Loading and Initial Analysis

The dataset is loaded using Pandas (pd.read_csv).

Basic analysis includes inspecting the dataset shape, information, summary statistics, and checking for missing or duplicated values.

2. Exploratory Data Analysis (EDA)

Distribution of numeric variables is visualized using histograms.

Outliers are identified using boxplots.

Relationships between features and the target variable (Life expectancy) are visualized using scatter plots.

A correlation heatmap is created to analyze feature correlations.

3. Data Preprocessing

Missing Value Treatment:

Missing values in specific columns (BMI, Polio, Income composition of resources) are filled with their median values.

The KNNImputer is used for imputing missing values in numeric columns.

Outlier Treatment:

Outliers in GDP, Total expenditure, and thinness 1-19 years are handled using the interquartile range (IQR) method.

Post-treatment, boxplots confirm the absence of outliers.

Categorical to Numerical Conversion:

Categorical columns are encoded using LabelEncoder.

Normalization:

Numeric columns are normalized using MinMaxScaler.

4. Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce the dataset's dimensionality to 5 components.

5. Splitting Dataset

The dataset is split into training and testing sets using an 80-20 ratio (train_test_split).

6. Model Evaluation

Four regression models are evaluated:

Random Forest Regressor

Support Vector Regressor (SVR)

Decision Tree Regressor

Linear Regression

Performance metrics:

Mean Squared Error (MSE)

R^2 Score

A Python function, evaluate_regression_models, trains and evaluates these models, printing the results for comparison.
