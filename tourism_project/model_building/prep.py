import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
#df.drop(columns=["CustomerID", "Unnamed"], inplace=True)
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True, errors="ignore")

# -----------------------------
# Feature Engineering
# -----------------------------

df["FamilySize"] = (
    df["NumberOfPersonVisiting"] +
    df["NumberOfChildrenVisiting"]
)
df["IncomePerPerson"] = (
    df["MonthlyIncome"] /
    (df["NumberOfPersonVisiting"] + 1)
)
df["FrequentTraveller"] = (
    df["NumberOfTrips"] >= 4
).astype(int)
df["InternationalTraveller"] = (
    (df["Passport"] == 1) &
    (df["NumberOfTrips"] >= 2)
).astype(int)

# -----------------------------
# Cap Outliers
# -----------------------------

outlier_cols = [
    "Age",
    "MonthlyIncome",
    "DurationOfPitch",
    "NumberOfTrips",
    "NumberOfFollowups"
]
for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
#print("Type values kept as:", sorted(X["Type"].unique()))
