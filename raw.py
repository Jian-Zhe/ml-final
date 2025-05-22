# Importing libraries and setting up the environment
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensuring the backend for non-interactive environments
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # switch backend if only plt is imported
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Ensure inline plotting on Kaggle
# %matplotlib inline

print('Libraries imported and configured.')

# Load the dataset from the CSV file
data_path = 'heart-attack-risk-prediction-dataset.csv'
df = pd.read_csv(data_path, delimiter=',', encoding='ascii')

# Display basic information about the data
print('Dataset loaded.')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())

# Quick inspection of the data
print('\nFirst five rows:')
print(df.head())

print('\nData Types:')
print(df.info())

print('\nMissing values per column:')
print(df.isnull().sum())

###########################################################
from sklearn.impute import SimpleImputer

# Create a copy of the dataframe for preprocessing
df_clean = df.copy()

# Identify numeric columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Instantiate the simple imputer for numeric values
imputer = SimpleImputer(strategy='mean')

# Impute missing values for numeric features
df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

# If required, additional processing for categorical data can be done here

print('Missing values after imputation:')
print(df_clean.isnull().sum())

###############################################
# Correlation heatmap for numeric features (if there are four or more)
numeric_df = df_clean.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12,10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()
else:
    print('Less than 4 numeric features. Skipping heatmap.')

# Pair plot for a subset of numeric features (choose a few to avoid overcrowding)
selected_features = numeric_df.columns[:5]
sns.pairplot(df_clean[selected_features])
plt.suptitle('Pair Plot of Selected Numeric Features', y=1.02)
plt.show()

# Categorical analysis: showing count distribution for 'Gender'
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df_clean, palette='viridis')
plt.title('Distribution of Gender')
plt.show()

# Bar plot for 'Diabetes' counts (if it is categorical binary indicator)
plt.figure(figsize=(6,4))
sns.countplot(x='Diabetes', data=df_clean, palette='magma')
plt.title('Distribution of Diabetes')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define target variable and features
# We'll use "Heart Attack Risk (Binary)" as the target.
target = 'Heart Attack Risk (Binary)'

# Drop target and other non-numeric or irrelevant columns from the feature set
cols_to_drop = ['Heart Attack Risk (Binary)', 'Heart Attack Risk (Text)', 'Gender']
X = df_clean.drop(columns=cols_to_drop, errors='ignore')
y = df_clean[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the RandomForestClassifier. Default hyperparameters are used for demonstration.
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)
print('Model training complete.')

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate the prediction accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Prediction Accuracy:', accuracy)

print('feature importances: ', rf_model.feature_importances_)