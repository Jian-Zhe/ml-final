import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, rotation=rotation, label_type=location)
    axes.set_ylabel("")
    axes.set_xlabel("")
    axes.set_yticklabels(())

def training_classification():

    rfc = RandomForestClassifier()

    models_cl = [rfc]

    names_cl = ["Random Forest"]

    scores, reports, cms = [], dict(), dict()

    for i, j in zip(models_cl, names_cl):
        i.fit(x_train, y_train)
        pred = i.predict(x_test)
        scores += [accuracy_score(pred, y_test)*100]
        reports[j] = classification_report(pred, y_test)
        cms[j] = confusion_matrix(pred, y_test)

    fig, axes = plt.subplots()
    dd = pd.DataFrame({"score": scores}, index=names_cl)
    dd = dd.sort_values("score", ascending=False)
    dd["score"] = round(dd["score"], 2)
    dd["score"].plot(kind="bar", ax=axes)
    bar_labels(axes)
    plt.tight_layout()
    plt.show()

    index = 0

    # for _ in range(2):
    #     fig, axes = plt.subplots(ncols=4, figsize=(15, 6))
    #     for i in range(4):
    #         sns.heatmap(cms[dd.index[index]], annot=True, fmt='d', ax=axes[i])
    #         axes[i].set_title("{}: {}%".format(dd.index[index], dd.iloc[index, 0]))
    #         index += 1
    #     plt.tight_layout()
    #     plt.show()

    for i in dd.index:
        print("*"*30)
        print(i)
        print(reports[i])

        print("\n\n")

df = pd.read_csv("heart-attack-risk-prediction-dataset.csv")

df["Gender"] = df["Gender"].map({"Male": "Male", "Female": "Female", "1.0": "Male", "0.0": "Female"})

df["Heart Attack Risk (Text)"] = df["Heart Attack Risk (Text)"].apply(lambda x: 1 if x>=1 else 0)

feats = df.columns[:-4].tolist()
feats += df.columns[-3:].tolist()

feats += [df.columns[-4]]

df = df[feats]

cats = [i for i in df.columns if df[i].nunique() <= 4]
nums = [i for i in df.columns if i not in cats]

for i in cats:
    df[i] = df[i].fillna(df[i].mode()[0])

for i in nums:
    df[i] = df[i].fillna(df[i].median())

# index = 0

# for j in [6, 5]:
#     fig, axes = plt.subplots(ncols=j, figsize=(15, 6))
#     for i in range(j):
#         df[cats[index]].value_counts().plot(kind="bar", ax=axes[i])
#         bar_labels(axes[i])
#         axes[i].set_title(cats[index].replace('_', ' '))
#         index += 1
#     plt.tight_layout()
#     plt.show()

# index = 0

# for _ in range(4):
#     fig, axes = plt.subplots(ncols=4, figsize=(15, 6))
#     for i in range(4):
#         sns.histplot(df, x=nums[index], kde=True, ax=axes[i])
#         axes[i].set_xlabel("")
#         axes[i].set_ylabel("")
#         axes[i].set_title(nums[index].replace('_', ' '))
#         index += 1
#     plt.tight_layout()
#     plt.show()

for i in cats[:-1]:
    df[i] = LabelEncoder().fit_transform(df[i].values)

# wrong!!!
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# wrong end

# should use below
# target_col = "Heart Attack Risk (Binary)"
# x = df.drop(columns=[target_col, "Heart Attack Risk (Text)"])
# y = df[target_col]
# end

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

x_train, y_train = SMOTE().fit_resample(x_train, y_train)

training_classification()