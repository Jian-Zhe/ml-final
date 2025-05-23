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

# print(df.head())
# print(df.columns.tolist())
# exit(0)
# print(df.info())

# print(df.describe())

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
X = df.drop(columns=['Heart Attack Risk (Binary)', 'Heart Attack Risk (Text)',
                     "Cholesterol", "Triglycerides", "CK-MB", "Troponin",
                     "Gender_Female", "Family History"])
y = df[target_column]

random_state = 42

# Splitting Data
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Apply Hybrid Oversampling & Undersampling (SMOTETomek) to handle class imbalance
smote_tomek = SMOTETomek(random_state=random_state)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

feature_names = X.columns.tolist()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    # n_estimators=800,
    random_state=random_state,
    # min_samples_split=6,
    # min_samples_leaf=1,
    # max_depth=None,
    # bootstrap=False,
    # class_weight=None,
    # max_features="sqrt"
)

# param_grid = {
#     'n_estimators': [800],
#     'max_features': ['sqrt', 'log2', 0.3, None],
#     'bootstrap': [True, False],
#     'class_weight': [None, 'balanced'],
#     'min_samples_split': [6],
#     'min_samples_leaf': [1]
# }

# random_search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)
# random_search.fit(X_train, y_train)
# print(random_search.best_params_)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))


# Get feature importance
# importances = rf.feature_importances_

# Create DataFrame for better visualization
# feat_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

# print(feat_importance_df)

# result = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42)
# sorted_idx = result.importances_mean.argsort()[::-1]
# sorted_features = [feature_names[i] for i in sorted_idx]

# plt.figure(figsize=(12, 6))
# plt.bar(range(len(sorted_idx)), result.importances_mean[sorted_idx],
#         yerr=result.importances_std[sorted_idx])
# plt.xticks(range(len(sorted_idx)), sorted_features, rotation=90)
# plt.title("Permutation Feature Importance")
# plt.ylabel("Mean Decrease in Accuracy")
# plt.tight_layout()
# plt.show()