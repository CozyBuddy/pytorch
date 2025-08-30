import pandas as pd

df1 = pd.read_csv('서울시 대기질 자료 제공_2008-2011.csv' ,encoding='euc-kr')
df2 = pd.read_csv('서울시 대기질 자료 제공_2012-2015.csv' ,encoding='euc-kr')
df3 = pd.read_csv('서울시 대기질 자료 제공_2016-2019.csv' ,encoding='euc-kr')
df4 = pd.read_csv('서울시 대기질 자료 제공_2020-2021.csv' ,encoding='euc-kr')
df5 = pd.read_csv('서울시 대기질 자료 제공_2022.csv' ,encoding='euc-kr')

print(df1.info()) 
print(df2.info()) 
print(df3.info()) 
print(df4.info()) 
print(df5.info()) 

df5['초미세먼지(PM25)'] = df5['초미세먼지(PM2.5)']

df5.drop(['초미세먼지(PM2.5)'] , axis=1 ,inplace=True)
df = pd.concat([df1,df2,df3,df4,df5])

print(df.head(50))
print(df.shape) # 3000000데이터
print(df.info()) 

df = df.dropna()

print(df.isna().sum())

print(df.head())

print(df['구분'].unique())

import lightgbm as lgb

ldf = int(len(df)*0.99)


df['일시'] = pd.to_datetime(df['일시'])

df['월'] = df['일시'].dt.month

df['년'] = df['일시'].dt.year

df['일'] = df['일시'].dt.day

df = df.drop(['일시','미세먼지(PM10)'],axis=1)
df = df.reset_index(drop=True)

cond = df['구분'] == '평균'
cond2 = (df['초미세먼지(PM25)'] < 300 )& (df['초미세먼지(PM25)'] >0)
df = df[~cond]
df = df[cond2]

train = df.iloc[:ldf]

test = df.iloc[ldf:]

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.histplot(train['초미세먼지(PM25)'], bins=30, kde=True, color='skyblue')
# plt.title('초미세먼지(PM2.5) 분포 및 밀도')
# plt.xlabel('농도 (μg/m³)')
# plt.show()

target = train.pop('초미세먼지(PM25)')
train['구분'] = train['구분'].astype('category')
test['구분'] = test['구분'].astype('category')




print(train.head())

from sklearn.model_selection import train_test_split

X_train ,X_val , y_train, y_val = train_test_split(train,target, test_size=0.01 , random_state=0)

import numpy as np
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
model = lgb.LGBMRegressor( 
   num_leaves=100,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=3000,
    subsample=0.8,
    colsample_bytree=1.0,  
    feature_fraction=1.0, 
    random_state=42)
# 파라미터 개선

model.fit(X_train,y_train_log, eval_set=[(X_val,y_val_log)], categorical_feature=['구분'] )

y_pred_log = model.predict(X_val)

pred = np.expm1(y_pred_log)
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_val,pred)

from sklearn.metrics import r2_score

r2score = r2_score(y_val,pred)
print('rmse' , rmse)
print('r2score' , r2score)


import joblib

joblib.dump(model , 'mymodel.pkl')

