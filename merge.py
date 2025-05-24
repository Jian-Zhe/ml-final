import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========== 讀取資料 ==========
data1 = pd.read_csv("heart_attack_prediction_dataset.csv")
data2 = pd.read_csv("heart_attack_prediction_india.csv")

# ========== 前處理：data1 和 data2 對齊格式 ==========
def diet_score_to_category(score):
    if score >= 7:
        return 0  # Healthy
    elif score >= 4:
        return 1  # Average
    else:
        return 2  # Unhealthy

# Diet
data2["Diet"] = data2["Diet_Score"].apply(diet_score_to_category)
data1["Diet"] = data1["Diet"].map({"Healthy": 0, "Average": 1, "Unhealthy": 2})

# Gender
data1["Gender"] = data1["Sex"].map({"Female": 0, "Male": 1})
data2["Gender"] = data2["Gender"].map({"Female": 0, "Male": 1})

# Blood Pressure 拆分
if "Blood Pressure" in data1.columns:
    bp_split = data1["Blood Pressure"].str.split("/", expand=True)
    data1["Systolic blood pressure"] = pd.to_numeric(bp_split[0], errors='coerce')
    data1["Diastolic blood pressure"] = pd.to_numeric(bp_split[1], errors='coerce')

# 建立 data2_renamed 統一格式
data2_renamed = pd.DataFrame({
    "Age": data2["Age"],
    "Heart rate": np.nan,
    "Diabetes": data2["Diabetes"],
    "Family History": data2["Family_History"],
    "Smoking": data2["Smoking"],
    "Obesity": data2["Obesity"],
    "Alcohol Consumption": data2["Alcohol_Consumption"],
    "Exercise Hours Per Week": np.nan,
    "Diet": data2["Diet"],
    "Previous Heart Problems": data2["Heart_Attack_History"],
    "Medication Use": np.nan,
    "Stress Level": data2["Stress_Level"],
    "Sedentary Hours Per Day": np.nan,
    "Income": data2["Annual_Income"],
    "BMI": np.nan,
    "Physical Activity Days Per Week": data2["Physical_Activity"],
    "Sleep Hours Per Day": np.nan,
    "Heart Attack Risk (Binary)": data2["Heart_Attack_Risk"],
    "Blood sugar": np.nan,
    "Gender": data2["Gender"],
    "Systolic blood pressure": data2["Systolic_BP"],
    "Diastolic blood pressure": data2["Diastolic_BP"]
})

# 調整 data1 欄位
data1['Heart rate'] = data1['Heart Rate']
data1['Heart Attack Risk (Binary)'] = data1['Heart Attack Risk']
data1 = data1.drop(columns=['Patient ID', 'Sex', 'Cholesterol', 'Blood Pressure', 'Triglycerides', 
                            'Country', 'Continent', 'Hemisphere', 'Heart Attack Risk', 'Heart Rate'])

# 對齊欄位順序
data2_aligned = data2_renamed[data1.columns]

# ========== 拆分 data1 的訓練與測試集（保留自然分布） ==========
train_data1, test_data1 = train_test_split(data1, test_size=0.1, stratify=data1["Heart Attack Risk (Binary)"], random_state=42)

# ========== 合併 data2 到 train_data1 ==========
merge_ratio = 0.3
n_target = int(len(train_data1) * merge_ratio)
ratio_1 = 0.5
n_1 = int(n_target * ratio_1)
n_0 = n_target - n_1

sample_1 = data2_aligned[data2_aligned["Heart Attack Risk (Binary)"] == 1].sample(n=n_1, random_state=42)
sample_0 = data2_aligned[data2_aligned["Heart Attack Risk (Binary)"] == 0].sample(n=n_0, random_state=42)

train_combined = pd.concat([train_data1, sample_0, sample_1], ignore_index=True)

# ========== 補缺值 ==========
for df in [train_combined, test_data1]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)


# ========== 確認結果 ==========
print("Train 資料 shape：", train_combined.shape)
print("Test 資料 shape：", test_data1.shape)

print("\nTrain 類別分布：")
print(train_combined["Heart Attack Risk (Binary)"].value_counts())

print("\nTest 類別分布：")
print(test_data1["Heart Attack Risk (Binary)"].value_counts())

# ========== 儲存資料 ==========
train_combined.to_csv("train_combined.csv", index=False)
test_data1.to_csv("test_data.csv", index=False)
