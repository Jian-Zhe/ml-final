import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance

df = pd.read_csv("heart-attack-risk-prediction-dataset.csv")

df = df.copy()
for column in df.columns:
    if df[column].dtype == 'object':  # Categorical columns
        df[column] = df[column].fillna(df[column].mode()[0])
    else:  # Numeric columns
        df[column] = df[column].fillna(df[column].median())

# Convert categorical variables to numeric (if any)
df = pd.get_dummies(df, drop_first=True)

# print("Dataset Columns:", df.columns.tolist())

target_column = "Heart Attack Risk (Binary)"  # Use binary column for ML
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found in dataset. Please check column names.")

## "Cholesterol", "Triglycerides", "CK-MB", "Troponin", "Smoking"
X = df.drop(columns=['Heart Attack Risk (Binary)', 'Heart Attack Risk (Text)'])
y = df[target_column]

random_state = 42

# Splitting Data
test_size = 0.1

# Apply Hybrid Oversampling & Undersampling (SMOTETomek) to handle class imbalance
smote_tomek = SMOTETomek(random_state=random_state)
X, y = smote_tomek.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


feature_names = X.columns.tolist()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    random_state=random_state,
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
