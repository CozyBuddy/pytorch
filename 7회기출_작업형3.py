import pandas as pd

df = pd.read_csv('clam.csv')

print(df)

train = df.iloc[:210]

test = df.iloc[210:]

from statsmodels.formula.api import logit

model = logit('gender ~ weight',train).fit()

import numpy as np
print(np.exp(model.params['weight']))

#2
model = logit('gender ~ age + length + diameter + height + weight',train).fit()

print(model.llf * -2)

#3

model = logit('gender ~ weight',df).fit()

pred = model.predict(test)

print(pred)


#2-1

df = pd.read_csv('system_cpu.csv')

print(df.head())

#1

print(df.corr())
print(round(df.corr().iloc[0,3],3))

#2

cond = df['CPU'] < 100

df = df[cond]

from statsmodels.formula.api import ols

model = ols('ERP ~ Feature1 + Feature2 + Feature3 + CPU' ,df).fit()

print(round(model.rsquared,3))

#3
print(round(model.pvalues.sort_values(ascending=False).iloc[0],3))