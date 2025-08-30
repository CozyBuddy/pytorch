import pandas as pd

df = pd.read_csv('members.csv')

print(len(df))

df = df.dropna()

ldf = int(len(df) * 0.7)

df = df.iloc[:ldf]

print(len(df))

print(int(df['f1'].quantile(0.25)))

#2

df = pd.read_csv('year.csv',index_col='Unnamed: 0')

cond = df.index ==2000

df = df[cond]

df2 = df.T

means = df2[2000].mean()

cond = df2 > means
print(sum(df2[2000] > means))

#3

df = pd.read_csv('members.csv')

df = df.isna().sum()
print(df.sort_values(ascending=False).index[0])

##2

train = pd.read_csv('train_7.csv')

test = pd.read_csv('test_7.csv')

print(train.info())
print(test.info())

print(train.isna().sum())
print(test.isna().sum())

target = train.pop('TravelInsurance')

lt = len(train)

data = pd.concat([train,test])

data_o = pd.get_dummies(data)

train = data_o.iloc[:lt]

test = data_o.iloc[lt:]

print(train,test)

from sklearn.model_selection import train_test_split

X_train, X_val , y_train, y_val = train_test_split(train,target , test_size=0.02 , random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train,y_train)

y_pred = rf.predict_proba(X_val)


from sklearn.metrics import roc_auc_score

ras = roc_auc_score(y_val,y_pred[:,1])


print('ras' , ras)

pred = rf.predict_proba(test)

submit = pd.DataFrame({
    'index' : test.index,
    'y_pred' : pred[:,1]
})

submit.to_csv('result.csv' , index=False )

