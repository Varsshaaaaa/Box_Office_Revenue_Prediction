
# Box Office Revenue Prediction Using Linear Regression

## Overview
This project predicts box office revenue using machine learning techniques, particularly XGBoost regression. The dataset includes movie-related attributes such as MPAA ratings, genres, number of opening theaters, and release days.

## Dataset
- The dataset is loaded from `boxoffice.csv`.
- Irrelevant columns such as `world_revenue` and `opening_revenue` are removed.
- Categorical data (`MPAA`, `genres`) is handled using encoding techniques.
- Missing values are either dropped or filled using appropriate strategies.
- Numerical data is transformed and scaled for better performance.

## Steps in the Project

### 1. **Importing Libraries**
- Uses `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost`.
- Suppresses unnecessary warnings.

### 2. **Data Preprocessing**
- Loads the dataset and performs exploratory data analysis (EDA).
- Handles missing values by dropping or imputing them.
- Converts text-based categorical data into numerical representations.
- Log transformation is applied to normalize skewed distributions.

### 3. **Exploratory Data Analysis (EDA)**
- Visualizes MPAA rating distribution using a count plot.
- Plots histograms and boxplots for key features to understand their distribution and detect outliers.

### 4. **Feature Engineering**
- Converts text data (`genres`) into numerical form using `CountVectorizer`.
- Scales numerical features using `StandardScaler`.

### 5. **Splitting Data**
- Data is split into training and testing sets using an 80-20 ratio.
- `train_test_split()` is used with `random_state=42` for reproducibility.

### 6. **Model Training**
- XGBoost Regression model is used for predicting box office revenue.
- The model is trained with 100 estimators and a learning rate of 0.1.

### 7. **Model Evaluation**
- Predictions are compared with actual values.
- Performance metrics used:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- A scatter plot is generated to visualize actual vs predicted revenue values.

## Results
- The model effectively predicts box office revenue based on the given features.
- Log transformation helps in improving the distribution of skewed data.
- The XGBoost model performs well compared to linear regression in handling non-linear relationships.

## How to Run the Project
1. Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn xgboost`
2. Load the dataset: `df = pd.read_csv('boxoffice.csv', encoding='latin-1')`
3. Preprocess data by handling missing values and transforming features.
4. Split data into training and testing sets.
5. Train the model using XGBoost.
6. Evaluate the model performance using metrics and visualization.

## Key Takeaways
- Data preprocessing significantly impacts model performance.
- Feature scaling and transformation improve model accuracy.
- Advanced regression models like XGBoost perform better than simple linear regression.

This README provides a structured explanation for HR discussions and technical presentations.

