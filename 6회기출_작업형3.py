import pandas as pd

df = pd.DataFrame({
    '항암약' : [4,4,3,4,1,4,1,4,1,4,4,2,1,4,2,3,2,4,4,4]
})

print(df)

from scipy.stats import chisquare

cond = df['항암약'] == 1
cond2 = df['항암약'] == 2
cond3 = df['항암약'] == 3
cond4 = df['항암약'] == 4
#1

nop = len(df[cond4]) / (len(df[cond])+len(df[cond2])+len(df[cond3])+len(df[cond4]))
#2

print(nop)
df = [len(df[cond]),len(df[cond2]),len(df[cond3]),len(df[cond4])]
df2 = [20*0.1 , 20*0.05 , 20*0.15 , 20*0.7]
print(chisquare(df,df2))
#6.976190476190476
#3
#0.07266054733847571


df = pd.read_csv('data6-3-2.csv')

print(df.head())

from statsmodels.formula.api import ols

model = ols('temperature ~ solar+wind+o3', df).fit()

print(model.summary())

#1

# 0.0749 

#2
from statsmodels.stats.anova import anova_lm

print(anova_lm(model))
print(model.pvalues['wind'])

#3

cond3 = pd.DataFrame({
    'solar' : [100],
    'wind' : [5] ,
    'o3' :[30]
})

print(model.predict(cond3)[0])