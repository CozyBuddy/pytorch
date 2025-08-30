import pandas as pd

train = pd.read_csv('train_1.csv')
test = pd.read_csv('test.csv')

print(train.head())

print(train.shape)
print(test.shape)

print(train.info())

print(train.describe(include='O'))

print(train.isna().sum())
print(test.isna().sum())

target = train.pop('Credit_Score')

data = pd.concat([train,test])
ntrain = len(train)
data_o = pd.get_dummies(data)

train = data_o.iloc[:ntrain]
test = data_o.iloc[ntrain:]

from sklearn.model_selection import train_test_split

X_train , X_val , y_train ,y_val= train_test_split(train,target , test_size=0.2 , random_state=0)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_val)

from sklearn.metrics import f1_score

f1 = f1_score(y_pred,y_val , average='macro')

print('f1' , f1)

pred = rf.predict(test)
submit = pd.DataFrame({
    'pred' : pred
})

submit.to_csv('result.csv',index=False)

train = pd.read_csv('diabetes_train.csv')
test = pd.read_csv('diabetes_test.csv')

print(train.head())
print(test.head())

target = train.pop('Outcome')

data = pd.concat([train,test])

data_o = pd.get_dummies(data)

ltrain = len(train)

train = data_o.iloc[:ltrain]
test = data_o.iloc[ltrain:]

from sklearn.model_selection import train_test_split

X_train, X_val , y_train ,y_val = train_test_split(train,target , test_size=0.2 , random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict_proba(X_val)

from sklearn.metrics import roc_auc_score

rauc = roc_auc_score(y_val,y_pred[:,1])

print('rocauc' ,rauc)

pred = rf.predict_proba(test)

submit = pd.DataFrame({''
'pred':pred[:,1]})

submit.to_csv('result.csv',index=False)

train = pd.read_csv('hr_train.csv')
test = pd.read_csv('hr_test.csv')

print(train)
print(test)

target = train.pop('target')

ntrain = len(train)

data = pd.concat([train,test])

data_o = pd.get_dummies(data)

print(train.isna().sum())
print(test.isna().sum())

train = train.fillna('X')
test = test.fillna('X')
train = data_o.iloc[:ntrain]
test = data_o.iloc[ntrain:]

from sklearn.model_selection import train_test_split
X_train,X_val ,y_train,y_val = train_test_split(train,target , test_size=0.1 , random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict_proba(X_val)

from sklearn.metrics import roc_auc_score

rac = roc_auc_score(y_val,y_pred[:,1])

print('rac', rac)

pred = rf.predict_proba(test)
submit = pd.DataFrame({
    'pred' : pred[:,1]
})

submit.to_csv('result.csv', index=False)


train = pd.read_csv('creditcard_train.csv')
test = pd.read_csv('creditcard_test.csv')



train['OCCUPATION_TYPE'] = train['OCCUPATION_TYPE'].fillna(train['OCCUPATION_TYPE'].mode()[0])

target = train.pop('STATUS')

data = pd.concat([train,test])

data_o = pd.get_dummies(data)

ntrain = len(train)

train = data_o.iloc[:ntrain]
test = data_o.iloc[ntrain:]

from sklearn.model_selection import train_test_split

X_train , X_val , y_train , y_val = train_test_split(train,target , test_size=0.2 , random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train ,y_train)

y_pred = rf.predict(X_val)

from sklearn.metrics import f1_score

f1s = f1_score(y_val , y_pred)

print(f1s)

submit = pd.DataFrame({
    'pred' : y_pred
})

submit.to_csv('result.csv' , index=False)



