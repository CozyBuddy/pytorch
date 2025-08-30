import os 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset ,DataLoader
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split

data1 = pd.read_csv('서울시 대기질 자료 제공_2008-2011.csv' , encoding='euc-kr')
data2 = pd.read_csv('서울시 대기질 자료 제공_2012-2015.csv', encoding='euc-kr')
data3 = pd.read_csv('서울시 대기질 자료 제공_2016-2019.csv' , encoding='euc-kr')
data4 = pd.read_csv('서울시 대기질 자료 제공_2020-2021.csv' , encoding='euc-kr')
data5 = pd.read_csv('서울시 대기질 자료 제공_2022.csv' , encoding='euc-kr')
data6 = pd.read_csv('서울시 대기질 자료 제공_2023.csv' , encoding='euc-kr')
data7 = pd.read_csv('서울시 대기질 자료 제공_2024.csv' , encoding='euc-kr')


data5['초미세먼지(PM25)'] = data5['초미세먼지(PM2.5)']

data5.drop(columns=['초미세먼지(PM2.5)'],inplace=True)
cond = data1['구분'] == '평균'
cond2 = data2['구분'] == '평균'
cond3 = data3['구분'] == '평균'
cond4 = data4['구분'] == '평균'
cond5 = data5['구분'] == '평균'
cond6 = data6['구분'] == '평균'
cond7 = data7['구분'] == '평균'

data1 = data1[cond]
data2 = data2[cond2]
data3 = data3[cond3]
data4 = data4[cond4]
data5 = data5[cond5]
data6 = data6[cond6]
data7 = data7[cond7]
df = pd.concat([data1,data2,data3,data4,data5,data6,data7])
df2 = pd.concat([data1,data2,data3,data4,data5,data6,data7])

print(df.describe())

## 평균 데이터만 갯수를 세서 약 15만건의 데이터를 확보하였고 
## 데이터의 값들은 사진과 같고

print(df.isnull().sum()) ## 결측치는 존재X

print(df.corr(numeric_only=True)) ## 초미세먼지와 미세먼지의 상관계수는 0.748 정도로 높음

df['일시'] =  pd.to_datetime( df['일시'])
df['일시'] = df['일시'].dt.date
df = df.groupby(df['일시']).mean(numeric_only=True)

print(df)
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
# plt.figure(figsize=(6,12))
# plt.plot(df.index, df['초미세먼지(PM25)'])
# plt.title('미세먼지 추이')
# plt.xlabel('날짜')
# plt.ylabel('수치')

# plt.show() 
df2['일시'] = pd.to_datetime(df2['일시'])
#df2['일시'] = df2['일시'].dt.date
df2['년'] = df2['일시'].dt.year
#df2['월일'] = df2['일시'].dt.month.astype('str') +'-'+ df2['일시'].dt.day.astype('str') 
df2['월일'] = df2['일시'].dt.month 
plt.figure(figsize=(24,6))
for i in range(2008,2024):
    data = df2[df2['년'] == i]
    
    data = data.groupby(data['월일']).mean(numeric_only=True).reset_index()
    data = data.sort_values('월일')
    
    plt.plot(data['월일'], data['초미세먼지(PM25)'] , label=str(i))
    

plt.xlabel('월-일')
plt.ylabel('미세먼지 수치 (PM2.5)')
plt.title('년도별 월-일 미세먼지 수치 비교')
plt.legend(loc='upper right')
plt.grid(True)
#plt.show()
#print(df2.head())

import seaborn as sns

# 상관계수 행렬 계산
# corr = df2[['초미세먼지(PM25)', '월일']].corr()
# print(corr)

import statsmodels.api as sm
from statsmodels.formula.api import ols

# 월과 PM2.5 컬럼이 있는 DataFrame
model = ols('Q("초미세먼지(PM25)") ~ C(월일)', data=df2).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

