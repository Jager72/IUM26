import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("ihormuliar/starbucks-customer-data", output_dir=".\\Data\\")

#------------

dataset = pd.read_csv(".\Data\\profile.csv", index_col=0)
dataset["age"] = pd.to_numeric(dataset["age"], errors="coerce").astype(float)

#Gender
dataset["gender"] = dataset["gender"].fillna("Unknown")
dataset = pd.get_dummies(dataset, columns=["gender"])

#Age
ageMean = dataset["age"].mean(skipna=True)
ageSTD  = dataset["age"].std(skipna=True)

ageNa = dataset["age"].isna()
missingNum = ageNa.sum()
randAge = np.random.normal(loc=ageMean, scale=ageSTD, size=missingNum)
randAge = np.clip(randAge, 18, 100)
dataset.loc[ageNa, "age"] = randAge

ageMean = dataset["age"].mean()
ageSTD  = dataset["age"].std()

#z = (dataset["age"] - ageMean) / ageSTD
outliers = dataset["age"] > 100
randAge = np.random.normal(ageMean, ageSTD, outliers.sum())
randAge = np.clip(randAge, 18, 100)
dataset.loc[outliers, "age"] = randAge
dataset["age"] = MinMaxScaler().fit_transform(dataset[["age"]])

#income
dataset["income"] = dataset["income"].fillna(dataset["income"].median())
low, high = dataset["income"].quantile([0.01, 0.99])
dataset["income"] = dataset["income"].clip(low, high)
dataset["income"] = MinMaxScaler().fit_transform(dataset[["income"]])

dataset["became_member_on"] =pd.to_datetime(dataset["became_member_on"], format="%Y%m%d").astype("int64")
dataset["became_member_on"] = MinMaxScaler().fit_transform(dataset[["became_member_on"]])

#ID ZOSTAWIAM DO ŁĄCZENIA RECORDÓW W ZALEZNOŚCI OD ZADANIA!

dataset2 = pd.read_csv(".\Data\\portfolio.csv", index_col=0)


dataset2["reward"] = MinMaxScaler().fit_transform(dataset2[["reward"]])
dataset2["difficulty"] = MinMaxScaler().fit_transform(dataset2[["difficulty"]])
dataset2["duration"] = MinMaxScaler().fit_transform(dataset2[["duration"]])

dataset2 = pd.get_dummies(dataset2, columns=["offer_type"])
channels = pd.get_dummies(dataset2["channels"].explode()).groupby(level=0).sum()
dataset2 = dataset2.drop("channels", axis=1).join(channels)

#----

dataset3 = pd.read_csv(".\Data\\transcript.csv", index_col=0)
dataset3["time"] = MinMaxScaler().fit_transform(dataset3[["time"]])
dataset3 = pd.get_dummies(dataset3, columns=["event"])

import ast
dataset3["value"] = dataset3["value"].apply(lambda s: ast.literal_eval(s).get("offer id"))

# #-----

result = (
    dataset3
    .merge(dataset2, left_on="value", right_on="id", how="left")
    .merge(dataset, left_on="person", right_on="id", how="left")
)

result = result.drop(columns=["person","value"])

#---

train, test = train_test_split(result, test_size=0.2, random_state=67)
test, eval = train_test_split(test, test_size=0.5, random_state=67)
# sns.histplot(dataset["age"], bins=20, kde=True)
# plt.title("Age Distribution")
# plt.show()

print("Dataset:", result.shape, "")
print("train:", train.shape, "")
print("test:", test.shape, "")
print("eval:", eval.shape, "")

def statystyka(col : pd.Series):
    print(f"{col.name}: mean - {col.mean()} min - {col.min()}\n max - {col.max()} std - {col.std()}\n skew - {col.skew()} median - {col.median()}\n")

statystyka(result["income"])
statystyka(result["time"])
statystyka(result["age"])
statystyka(result["reward"])
statystyka(result["difficulty"])

print(result.columns)


numCols = ['reward', 'difficulty', 'duration', 'age', 'income', 'time']

result[numCols].hist(figsize=(12,8), bins=30)
plt.tight_layout()
plt.show()


binCo = [
'event_offer completed',
'event_offer received',
'event_offer viewed',
'event_transaction',
'offer_type_bogo',
'offer_type_discount',
'offer_type_informational',
'gender_F','gender_M','gender_O','gender_Unknown'
]

result[binCo].sum().sort_values().plot(
    kind='barh',
    figsize=(8,6),
    title="Counts of binary features"
)
plt.show()

plt.figure(figsize=(12,8))
corr = result.select_dtypes(include='number').corr()


sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title("Feature Correlation")
plt.show()