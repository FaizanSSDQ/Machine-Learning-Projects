# **California Housing Price Detection Project**

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Data Description](#data-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Models](#machine-learning-models)
   - [Linear Regression Models](#linear-regression-models)
   - [Tree-Based Models](#tree-based-models)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results and Insights](#results-and-insights)
9. [Technologies Used](#technologies-used)
10. [Conclusion](#conclusion)
11. [Contact Information](#contact-information)

---

## **Project Overview**
The **California Housing Price Detection** project aims to predict the median house prices for California districts based on features such as median income, housing age, and location data. The dataset, sourced from the California housing data, serves as a comprehensive benchmark for understanding the housing market dynamics in the state.

This project encompasses the entire **data science pipeline**, from **data cleaning** and **exploratory data analysis (EDA)** to **feature engineering** and building **machine learning models** for predictive analysis.

---

## **Key Features**
- **Data Exploration and Visualization**: Detailed exploration of the data to understand patterns, outliers, and correlations between different features.
- **Data Preprocessing**: Transforming data through scaling, one-hot encoding, and handling of missing values.
- **Model Comparison**: Comparing multiple regression models, including linear regression techniques and tree-based models, to identify the most accurate model.
- **Hyperparameter Tuning**: Optimizing model parameters to achieve better prediction accuracy.

---

## **Data Description**
The dataset contains the following features:
- **longitude**: A measure of how far west a house is; a higher value is farther west.
- **latitude**: A measure of how far north a house is; a higher value is farther north.
- **housingMedianAge**: Median age of houses in the district; a lower number indicates newer buildings.
- **totalRooms**: Total number of rooms within a block.
- **totalBedrooms**: Total number of bedrooms within a block.
- **population**: Total number of people residing within a block.
- **households**: Total number of households, defined as a group of people residing in one unit.
- **medianIncome**: Median income for households in a district (in tens of thousands of dollars).
- **medianHouseValue**: Target variable indicating the median house value for households (in U.S. dollars).
- **oceanProximity**: Categorical data indicating the location of the house with respect to the ocean.

---

## **Exploratory Data Analysis (EDA)**
Key insights from EDA include:
- **Correlation Analysis**: Heatmaps visualize correlations between continuous variables.
- **Distribution Plots**: Histograms, scatter plots, and boxplots to detect outliers.
- **Categorical Data Analysis**: The `oceanProximity` feature was analyzed using bar plots and one-hot encoding.

---

## **Data Preprocessing**
Key steps in data preprocessing:
1. **Handling Missing Values**: Missing values in the `totalBedrooms` feature were imputed with median values.
2. **One-Hot Encoding**: Categorical feature `oceanProximity` was converted to numerical format using one-hot encoding.
3. **Feature Scaling**: Continuous features were scaled using the `StandardScaler` to normalize the data for better model performance.

---

## **Machine Learning Models**

### **Linear Regression Models**
- **Linear Regression**: Baseline model to establish a benchmark.
- **Lasso Regression**: Reduces model complexity and performs feature selection.
- **Ridge Regression**: Prevents overfitting through L2 regularization.

### **Tree-Based Models**
- **Decision Tree Regressor**: Captures non-linear relationships between features.
- **Random Forest Regressor**: Ensemble of decision trees for improved accuracy and reduced variance.
- **Gradient Boosting Regressor**: Sequential trees correct errors from previous models.

---

## **Evaluation Metrics**
Models were evaluated using:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of prediction errors.
- **Mean Squared Error (MSE)**: Penalizes larger errors for sensitive evaluation.
- **R² Score**: Indicates how well the model explains variance in the target variable.

---

## **Results and Insights**
- **Linear Models**: Moderate performance with some overfitting in complex variants like Lasso and Ridge.
- **Tree-Based Models**: Random Forest and Gradient Boosting provided the best accuracy, capturing non-linear patterns in housing data.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **Pandas**: Data manipulation and analysis.
  - **NumPy**: Numerical computing.
  - **Scikit-learn**: Machine learning modeling and evaluation.
  - **Matplotlib & Seaborn**: Data visualization.

---

## **Conclusion**
This project demonstrates a complete pipeline for building and evaluating machine learning models to predict California housing prices. **Tree-based models** showed superior performance due to their ability to capture complex feature interactions. Future extensions could include further hyperparameter tuning, additional feature engineering, or integration with other data sources.

---

## **Contact Information**
Feel free to connect with me for any questions, collaboration opportunities, or feedback:
- **LinkedIn**: https://www.linkedin.com/in/faizan-saleem-siddiqui-4411bb247/
