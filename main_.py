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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=random_state,
    # min_samples_split=6,
    # min_samples_leaf=1,
    max_depth=None,
    bootstrap=False,
    # class_weight=None,
    # max_features="log2"
)
# Parameter search
# param_grid = {
#     'n_estimators': [100, 200, 300, 400],
#     'max_features': ['sqrt', 'log2', 0.3, None],
#     'bootstrap': [True, False],
#     # 'class_weight': [None, 'balanced'],
#     'min_samples_split': [2, 4, 6, 8],
#     'min_samples_leaf': [1, 2, 4]
# }

# random_search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)
# random_search.fit(X_train, y_train)
# print(random_search.best_params_)

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

################################################
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import f1_score
# import numpy as np

# # 取得 precision, recall, thresholds
# precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)

# # 計算 f1-score 對應每個 threshold（注意 thresholds 長度比 precision/recall 少 1）
# f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

# # 找出 F1-score 最大值及對應 threshold
# best_idx = np.argmax(f1_scores)
# best_threshold = thresholds[best_idx]
# best_f1 = f1_scores[best_idx]

# # 繪圖
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precision[:-1], label="Precision")
# plt.plot(thresholds, recall[:-1], label="Recall")
# plt.plot(thresholds, f1_scores, label="F1-score", linestyle="--", color="black")

# # 標出最佳 F1 分數的點
# plt.axvline(x=best_threshold, color="red", linestyle=":")
# plt.scatter(best_threshold, best_f1, color="red", zorder=5)
# plt.text(best_threshold, best_f1, f' Best F1={best_f1:.2f}\n@Threshold={best_threshold:.2f}', 
#          verticalalignment='bottom', horizontalalignment='right', fontsize=10, color='red')

# plt.xlabel("Threshold")
# plt.ylabel("Score")
# plt.title("Precision, Recall, and F1-score vs Threshold")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# feature search
# from sklearn.feature_selection import SequentialFeatureSelector

# sfs = SequentialFeatureSelector(rf, 
#                                 n_features_to_select=17,
#                                 direction='forward', 
#                                 scoring='accuracy', 
#                                 cv=5)
# sfs.fit(X_train, y_train)
# print("Selected features:", X.columns[sfs.get_support()])

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