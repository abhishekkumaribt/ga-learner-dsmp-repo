# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here

df = pd.read_csv(path)
df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].replace({'\$': '', ',': ''}, regex=True)
X = df.drop(columns="CLAIM_FLAG")
y = df["CLAIM_FLAG"]
count = y.value_counts()
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=6, test_size=0.3)

# Code ends here


# --------------
# Code starts here

X_train = X_train.astype({"INCOME" : float, "HOME_VAL" : float, "BLUEBOOK" : float, "OLDCLAIM" : float, "CLM_AMT" : float})
X_test = X_test.astype({"INCOME" : float, "HOME_VAL" : float, "BLUEBOOK" : float, "OLDCLAIM" : float, "CLM_AMT" : float})

# Code ends here


# --------------
# Code starts here

X_train = X_train.dropna(subset=['YOJ','OCCUPATION'])
X_test = X_test.dropna(subset=['YOJ','OCCUPATION'])
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
X_train[["AGE","CAR_AGE","INCOME", "HOME_VAL"]] = X_train[["AGE","CAR_AGE","INCOME", "HOME_VAL"]].fillna(X_train[["AGE","CAR_AGE","INCOME", "HOME_VAL"]].mean())
X_test[["AGE","CAR_AGE","INCOME", "HOME_VAL"]] = X_test[["AGE","CAR_AGE","INCOME", "HOME_VAL"]].fillna(X_test[["AGE","CAR_AGE","INCOME", "HOME_VAL"]].mean())

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here

for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Code ends here


# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

model = LogisticRegression(random_state=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
score

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
score

# Code ends here


