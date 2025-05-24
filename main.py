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
from ydata_profiling import ProfileReport

df = pd.read_csv("heart-attack-risk-prediction-dataset.csv")

# 產生報告
# profile = ProfileReport(df, title="Heart Attack Dataset Report", explorative=True)

# # 輸出成 HTML
# profile.to_file("heart_attack_data_profile.html")
# exit(0)

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
                     "Diastolic blood pressure"])
y = df[target_column]

random_state = 42

# Splitting Data
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Apply Hybrid Oversampling & Undersampling (SMOTETomek) to handle class imbalance
smote_tomek = SMOTETomek(random_state=random_state)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

# counts = y_train.value_counts()
# print(counts)
# exit(0)

feature_names = X.columns.tolist()

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=random_state,
    max_depth=None,
    bootstrap=False,
)

# evalution
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, f1_score

rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

y_pred_rf = (y_proba_rf >= 0.27).astype(int)

# 評估
print("F1-score:", f1_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Average Precision (PR AUC)
ap_score = average_precision_score(y_test, y_proba_rf)
print("Average Precision (PR AUC):", ap_score)


###############################################
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred_rf)

# 畫出混淆矩陣熱圖
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
