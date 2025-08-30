# import pandas as pd

# df = pd.read_csv('drinks.csv')

# print(df.info())

# df2 = df.groupby('continent')['beer_servings'].mean().sort_values(ascending=False)

# print(df2)
# print(df2.index[0])

# cond = df['continent'] == df2.index[0]
# print(df[cond].sort_values('beer_servings',ascending=False).iloc[4,1])

# df = pd.read_csv('tourist.csv')
# print(df.head())
# print(df.info())

# df['관광객비율'] = df['관광'] / (df['관광']+df['공무']+df['사업']+df['기타'])

# a = df.sort_values('관광객비율', ascending=False).iloc[1,3]
# print(df.sort_values('관광객비율', ascending=False).iloc[1,3])

# b = df.sort_values('관광', ascending=False).iloc[1,2]

# print(b)

# print(a+b)

# train = pd.read_csv('churn_train.csv')
# test = pd.read_csv('churn_test.csv')

# print(train , test)

# print(train.info())
# print(test.info())

# print(train.isna().sum())
# print(test.isna().sum())

# target = train.pop('TotalCharges')

# lt = len(train)
# data = pd.concat([train,test])

# data_o = pd.get_dummies(data)

# train = data_o.iloc[:lt]
# test = data_o.iloc[lt:]

# from sklearn.model_selection import train_test_split

# X_train, X_val , y_train ,y_val = train_test_split(train,target , test_size=0.1 , random_state=0)

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()

# rf.fit(X_train ,y_train)

# y_pred = rf.predict(X_val)

# from sklearn.metrics import mean_absolute_error
# mae = mean_absolute_error(y_val ,y_pred)

# print('mae' , mae)

# pred = rf.predict(test)
# submit = pd.DataFrame({
#     'pred' : pred
# })

# submit.to_csv('result.csv', index=False)

# print(pred.shape)
# print(test.shape)
# print(pd.read_csv('result.csv').head())

import pandas as pd
df = pd.read_csv('churn.csv')
print(df.info())
from statsmodels.formula.api import logit

model = logit('Churn ~ AccountWeeks+ContractRenewal+DataPlan+DataUsage+CustServCalls+DayMins+DayCalls+MonthlyCharge+OverageFee+RoamMins',df).fit()

cond= model.pvalues > 0.05
print(len(model.pvalues[cond]))
print(model.pvalues[cond])

model = logit('Churn ~ DataUsage+DayMins',df).fit()

print(round(sum(model.params),3))

import numpy as np
print(round(np.exp(model.params['DataUsage']*5) ,3))

df= pd.read_csv('piq.csv')

print(df.head())

print(df.info())

from statsmodels.formula.api import ols

model = ols('PIQ ~ Brain + Height + Weight' ,df).fit()

print(model.summary())

print(model.pvalues.sort_values().sort_values().iloc[0])

print(round(model.rsquared,2))

print(round(model.predict({
    'Brain' : [90] ,
    'Height' : [70],
    'Weight' : [150]
}).iloc[0],0))