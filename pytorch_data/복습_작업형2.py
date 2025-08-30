import pandas as pd

train = pd.read_csv('train_5.csv')
test = pd.read_csv('test_5.csv')

print(train.head())

print(train.shape)
print(test.shape)

print(train.info())
print(test.info())

print(train.describe(include='O'))
print(train.isna().sum())

print(train['income'].value_counts())

train = train.dropna()

print(train.isna().sum())

train = train[train['age'] >=1]
test = test[test['age'] >=1]

print(train.shape)
print(test.shape)

y_train = train.pop('income')

data = pd.concat([train,test])
lentrain = len(train)
data_oh = pd.get_dummies(data)

train = data_oh.iloc[:lentrain]
test = data_oh.iloc[lentrain:]

print(train.shape)
print(test.shape)

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(train,y_train ,test_size=0.2 ,random_state=0)

print(X_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

pred = rf.predict_proba(X_val)

print(pred)

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_val,pred[:,1])
print(roc_auc)

from sklearn.metrics import accuracy_score
pred = rf.predict(X_val)
acc = accuracy_score(y_val,pred)

print(acc)

from sklearn.metrics import f1_score
f1 = f1_score(y_val,pred,pos_label='>50K')
print(f1)

submit = pd.DataFrame({
    'pred' : pred
})

submit.to_csv('result.csv',index=False)

