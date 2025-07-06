import pandas as pd

df = pd.read_csv('data4-1.csv')

print(df.head())

print(int(abs(df['age'].quantile(0.75) - df['age'].quantile(0.25))))

#22

df= pd.read_csv('data4-2.csv')

print(df.head())

df['lwper'] = (df['loves'] + df['wows']) / (df['likes']+df['loves'] + df['wows']+ df['hahas'] + df['sads']+ df['angrys'])
cond = (df['lwper'] > 0.4 )& (df['lwper'] < 0.5)

cond2 = df['type'] == 'video'


print(len(df[cond & cond2]))

#3

df = pd.read_csv('data4-3.csv')

print(df.head())
print(df.info())

df['date_added'] = pd.to_datetime(df['date_added'])

print(df['date_added'].head())

cond = (df['date_added'].dt.year == 2018) & (df['date_added'].dt.month == 1)

cond2 = df['country'] == 'United Kingdom'
print(len(df[cond & cond2]))

train = pd.read_csv('train_8.csv')
test = pd.read_csv('test_8.csv')

print(train.info())
print(test.info())

target = train.pop('Segmentation')

lt = len(train)
data = pd.concat([train,test])

data_o = pd.get_dummies(data)

train = data_o.iloc[:lt]

test = data_o.iloc[lt:]

from sklearn.model_selection import train_test_split

X_train, X_val , y_train, y_val = train_test_split(train , target , test_size=0.01 , random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_val)

from sklearn.metrics import f1_score

f1 = f1_score(y_val,y_pred, average='macro')

print('f1' ,f1)

pred = rf.predict(test)
submit = pd.DataFrame({
    'ID' : test['ID'],
    'Segmentation' : pred
})

submit.to_csv('result.csv' , index=False)
