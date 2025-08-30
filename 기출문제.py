import pandas as pd

df = pd.read_csv('members.csv')

print(df.head())

df = df.sort_values('views' , ascending=False).reset_index()

print(df.head(10))

# 인덱스 0~8까지를 9번 인덱스 값으로 채우려면
df.loc[:8, 'views'] = df.loc[9, 'views']


print(df.head(10))

cond = df['age'] >= 80

print(round(df[cond]['views'].mean(),2))


df = pd.read_csv('members.csv')

ltrain = int(len(df) * 0.8)

df = df.iloc[:ltrain]

print(df.head())

print(df['f1'].var())

lvar = (df['f1'].var() ** 0.5)

mevalue = df['f1'].median()
df['f1'] = df['f1'].fillna(mevalue)

print(df['f1'].var())
rvar = df['f1'].var() ** 0.5

print(round(abs(rvar - lvar),2))

df = pd.read_csv('members.csv')

means = df['age'].mean()

vars = df['age'].var() ** 0.5

cond = (df['age'] > means + vars *1.5) | ( df['age'] < means - vars*1.5)

print(df[cond]['age'].sum())

####2


X_test = pd.read_csv('X_test.csv')

X_train = pd.read_csv('X_train.csv')

y_train = pd.read_csv('y_train.csv')

print(X_test.head())

print(X_train.info())
print(X_test.info())
print(y_train.info())


ltrain = len(X_train)
train = pd.concat([X_train,X_test])

X_train2 = pd.get_dummies(train)

X_train = X_train2.iloc[:ltrain]
X_test = X_train2.iloc[ltrain:]
print(y_train.head())

from sklearn.model_selection import train_test_split

X_train, X_val ,y_train,y_val = train_test_split(X_train,y_train['Reached.on.Time_Y.N'],test_size=0.1 , random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_val)

from sklearn.metrics import roc_auc_score

ras = roc_auc_score(y_val,y_pred)

print('ras',ras)

pred = rf.predict_proba(X_test)
submit = pd.DataFrame({''
'Reached.on.Time_Y.N' : pred[:,1] })

submit.to_csv('result.csv',index_label='ID')
