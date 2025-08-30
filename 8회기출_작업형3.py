import pandas as pd

df = pd.read_csv('churn.csv')

print(df.head())

print(df.info())

from statsmodels.formula.api import logit

model = logit('Churn ~ AccountWeeks + ContractRenewal +DataPlan+DataUsage+CustServCalls+DayMins+DayCalls+MonthlyCharge+OverageFee+RoamMins',df).fit()

df2 = model.pvalues

cond = df2 >= 0.05
#1
print(len(df2[cond]))
print(df2[cond])

#2
model = logit('Churn ~ DataUsage+DayMins',df).fit()

print(round(model.params.sum(),3))

#3

import numpy as np

print(np.exp(model.params['DataUsage'] *5))

df = pd.read_csv('piq.csv')

print(df.head())

#1
print(df.info())

from statsmodels.formula.api import ols

model = ols('PIQ ~ Brain + Height + Weight',df).fit()

print(model.pvalues.sort_values().index[0])

print(round(model.params['Brain'],3))

#2

print(round(model.rsquared,2))

#3

option = pd.DataFrame({
    'Brain' : [90],
    'Height' : [70],
    'Weight' : [150],
    
})

print(int(round(model.predict(option)[0],0)))
