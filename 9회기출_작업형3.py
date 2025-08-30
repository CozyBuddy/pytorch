import pandas as pd

df = pd.read_csv('loan.csv')

print(df.info())

df['총대출액'] = df['신용대출'] + df['담보대출']


df2 = df.groupby(['지역코드' ,'성별'])['총대출액'].sum().unstack()
print(df2.head())
df2['차이'] = abs(df2[1]- df2[2])
print(df2.head())

print(df2)

#2

df = pd.read_csv('crime.csv')

print(df.info())

print(df.head())


dflong = pd.melt(df, id_vars=['연도','구분'] , var_name='범죄유형' , value_name='건수')

print(dflong)

dfpivot = dflong.pivot_table(index=['연도','범죄유형'] , columns='구분' , values='건수')

print('피벗테이블 \n' ,dfpivot)

dfpivot['비율'] = dfpivot['검거건수'] / dfpivot['발생건수']

#dfpivot.groupby('연도')['비율'].idxmax()

print(dfpivot.loc[dfpivot.groupby('연도')['비율'].idxmax()]['검거건수'].sum())

df = pd.read_csv('hr.csv')

print(df.isna().sum())
m = df['만족도'].mean()
df['만족도'] = df['만족도'].fillna(m)

df2 = df.groupby(['부서','성과등급'])['근속연수'].transform('mean').astype('int')

df['근속연수'] = df['근속연수'].fillna(df2)

print(df.isna().sum())

df['계산한값'] = df['연봉'] / df['근속연수']

print(df.sort_values('계산한값',ascending=False))
print(df.sort_values('계산한값',ascending=False).iloc[2,4])

df['계산2'] = df['연봉'] / df['만족도']
print(df.sort_values('계산2',ascending=False))
print(df.sort_values('계산2',ascending=False).iloc[1,5])

print(2)
#df['근속연수'] = df['근속연수'].fillna()


train = pd.read_csv('farm_train.csv')

test = pd.read_csv('farm_test.csv')

print(train.info())
print(test.info())

print(train.isna().sum())
print(test.isna().sum())

print(train['농약검출여부'].value_counts())
target = train.pop('농약검출여부')

lt = len(train)

data = pd.concat([train,test])

data_o = pd.get_dummies(data)

train = data_o.iloc[:lt]

test = data_o.iloc[lt:]

from sklearn.model_selection import train_test_split

X_train,X_val , y_train ,y_val = train_test_split(train,target,test_size=0.05 , random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_val)

from sklearn.metrics import f1_score

f1 = f1_score(y_val,y_pred, average='macro')

print('f1' , f1)

pred = rf.predict(test)


submit = pd.DataFrame({
    'pred' : pred
})

submit.to_csv('result.csv', index=False)


#3

df = pd.read_csv('design.csv')

train = df.iloc[:140]

test = df.iloc[140:]

from statsmodels.formula.api import ols

model = ols('design ~ c1+c2+c3+c4+c5',train).fit()

df2 = model.pvalues
print(df2)
cond = (df2 < 0.05 )& (df2.index !='Intercept')
print(len(df2[cond]))
print(df2[cond])

model = ols('design ~ c1+c2+c4',train).fit()

result = model.predict(train)
print(result)

result2 = train['design'].corr(result)

print(round(result2,3))

result3 = model.predict(test)

from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(test['design'],result3)

print('rmse', round(rmse,3))

#3

df= pd.read_csv('retention.csv')

print(df.head())
print(df.info())
from statsmodels.formula.api import logit

model = logit('Churn ~ MonthlyCharges+CustomerTenure+HasPhoneService+HasTechInsurance',df).fit()

print(round(model.pvalues[model.pvalues.index=='MonthlyCharges'],3))

#2

print(df.head())
import numpy as np
result =np.exp(model.params['HasPhoneService'])

print(round(result,3))

#3

result2 = model.predict(df)

print(len(result2[result2 > 0.3]))