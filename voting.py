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
                     "Diastolic blood pressure"])
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

##########################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(
    max_iter=10000,
    random_state=42,
    class_weight='balanced',
    solver='saga',
    penalty='elasticnet',
    C=0.1,
    l1_ratio=0.3
)
logreg.fit(X_train, y_train)

# 預測與評估
y_pred = logreg.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# exit(0)
##########################
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 建立個別模型
# rf = RandomForestClassifier(
#     n_estimators=300,
#     random_state=42,
#     min_samples_split=6,
#     min_samples_leaf=1,
#     max_depth=None,
#     bootstrap=False,
#     class_weight=None,
#     max_features="log2"
# )

xgb = XGBClassifier(
    n_estimators=180,
    eval_metric="aucpr",
    random_state=42,
    learning_rate=0.2,
    reg_lambda=10,
    reg_alpha=1,
    # scale_pos_weight=1.2,
)


# Parameter search
# param_grid = {
#     'n_estimators': [300, 400, 500, 600],
#     'learning_rate': [0.5, 0.3, 0.1],
#     'min_child_weight': [1, 3],
#     'subsample': [0.6, 0.8, 1],
#     'gamma': [0.3, 0.5, 0.8],
#     'reg_lambda': [2, 3, 4, 5],
#     'reg_alpha': [0, 1],
#     'scale_pos_weight': [3, 4, 5]
    
# }

# random_search = RandomizedSearchCV(xgb, param_grid, n_iter=100, cv=5, n_jobs=-1, random_state=42)
# random_search.fit(X_train, y_train)
# print(random_search.best_params_)

# Voting Classifier（soft voting，需支持 predict_proba）
voting_clf = VotingClassifier(
    estimators=[('logistic', logreg), ('xgb', xgb)],
    voting='soft',
    weights=[1, 0.8]
)

# 訓練模型
# rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
# voting_clf.fit(X_train, y_train)

# 預測
# pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)
# pred_vote = voting_clf.predict(X_test)

# 比較結果
# print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))
# print(classification_report(y_test, pred_rf))

print("XGBoost Accuracy:", accuracy_score(y_test, pred_xgb))
print(classification_report(y_test, pred_xgb))

# print("Voting Classifier Accuracy:", accuracy_score(y_test, pred_vote))
# print(classification_report(y_test, pred_vote))
