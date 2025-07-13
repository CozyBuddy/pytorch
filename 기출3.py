import pandas as pd

df = pd.read_csv('members.csv')


df = df.dropna()


print(df.isna().sum())

ldf = int(len(df) * 0.7)

df2 = df.iloc[:ldf]

IQr1 = df2['f1'].quantile(0.25)

print(int(IQr1))

#222

df = pd.read_csv('year.csv', index_col='Unnamed: 0')

print(df.head())

means = df.loc[2000,:].mean()
print(means)
cond = df > means
print(len(df[cond].loc[2000,:].dropna()))

df = pd.read_csv('members.csv')
print(df.isna().sum())

df = df.isna().sum()

print(df.sort_values(ascending=False).index[0])
