# Heart Disease Prediction Project

## Overview
This project focuses on building a predictive model to classify the risk of heart disease based on patient data. The goal is to create an end-to-end machine learning pipeline, starting from data preprocessing and exploratory data analysis (EDA) to building and fine-tuning different machine learning models. The data used for this project includes clinical features related to heart health, such as blood pressure, cholesterol levels, heart rate, and more.

## Project Structure
- **data/**: Contains the heart disease dataset used in the project.
- **notebooks/**: Jupyter notebooks showcasing step-by-step EDA, data preprocessing, model building, and evaluation.
- **src/**: Python scripts for data processing and modeling.
- **README.md**: Overview and documentation of the project.

---

## Dataset Information
The dataset used for this project contains features related to heart health and patient demographics. Here is a list of the main features:

- `age`: Age of the patient.
- `sex`: Gender of the patient.
- `cp`: Chest pain type (0-3 categories).
- `trtbps`: Resting blood pressure (in mm Hg).
- `chol`: Cholesterol in mg/dl.
- `fbs`: Fasting blood sugar (> 120 mg/dl).
- `restecg`: Resting electrocardiographic results (0-2).
- `thalachh`: Maximum heart rate achieved.
- `oldpeak`: Depression induced by exercise.
- `slp`: Slope of the peak exercise ST segment.
- `caa`: Number of major vessels colored by fluoroscopy.
- `thall`: Thallium stress test result.
- `exng`: Exercise-induced angina (1 = yes, 0 = no).
- `output`: Target variable indicating the presence of heart disease.

---

## Workflow and Steps

### 1. Data Inspection
We started by loading the dataset to understand its structure using:
- `data.head()`, `data.info()`, and `data.describe()` to get an overview of the dataset, including data types, missing values, and statistical summaries.

### 2. Exploratory Data Analysis (EDA)
Comprehensive data analysis was performed to uncover insights and patterns:
- **Histograms and Boxplots**: We plotted histograms and boxplots for continuous features to analyze their distributions and detect outliers.
- **Correlation Analysis**: Using a heatmap, we analyzed the relationships between continuous features, revealing potential correlations with the target variable `output`.
- **Pairplots and Scatterplots**: Visualized relationships between features and the target variable for better understanding.

### 3. Data Preprocessing
- **Handling Missing Values**: No missing values were detected in the dataset.
- **Outlier Detection**: Outliers were identified using boxplots; however, due to the limited dataset size, they were retained to preserve data integrity.
- **Categorical Encoding**: Categorical variables were encoded using one-hot encoding.
- **Feature Scaling**: Continuous features were scaled using `RobustScaler` to minimize the effect of outliers and improve model performance.

### 4. Feature Engineering
Categorical and continuous features were separated to ensure appropriate processing steps, leading to optimized feature sets for modeling.

### 5. Machine Learning Model Building
Two primary types of models were built:

#### 5.1 Linear Classification Models
- **Support Vector Classifier (SVC)**
- **Logistic Regression**
- **SVC with Hyperparameter Tuning**

#### 5.2 Tree-Based Models
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

### 6. Model Evaluation
Models were evaluated based on their accuracy and other performance metrics using a test dataset. The results were compared through bar plots for visual clarity, showcasing the effectiveness of each model.

---

## Key Insights
- People with certain medical conditions, such as high thalachh (maximum heart rate achieved), have a higher risk of heart disease.
- The distribution plots and bivariate analysis offered valuable insights for feature relationships with the target variable.
- Tree-based models like Random Forest and Gradient Boosting performed well in predicting heart disease.

---

## Future Work
- **Hyperparameter Tuning**: Further optimization of tree-based models using GridSearchCV.
- **Cross-Validation**: Implementing cross-validation to ensure the robustness of the models.
- **Feature Engineering**: Exploring new feature transformations for better model performance.
- **Model Interpretability**: Utilizing techniques like SHAP values to understand model predictions.

---

## Requirements
This project requires Python 3.x and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch (if using deep learning models)

To install the required packages, run:
```bash
pip install -r requirements.txt
```
## Conclusion
This project demonstrates a comprehensive approach to predicting heart disease using machine learning. By combining thorough EDA, data preprocessing, and a variety of models, we achieved valuable insights and strong predictive performance, which can help guide medical decision-making processes.
