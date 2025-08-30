import pandas as pd

df= pd.read_csv('loan.csv')

print(df.head())

df['총대출액'] = df['신용대출'] + df['담보대출']

print(df.head())

print(df.groupby(['지역코드','성별']).sum())

df2 = df.groupby(['지역코드','성별']).sum()

df2 = df.pivot_table( index=['지역코드'] , columns=['성별'] , values='총대출액' , aggfunc='sum')
print(df2.head())
df2['차이'] = abs(df2[1] - df2[2])
print(df2.sort_values('차이',ascending=False).index[0])
# cond = df2['대출종류'] == '총대출액'
# print(df2[cond].head(10))

df = pd.read_csv('crime.csv')

print(df.head())

df = pd.melt(df , id_vars=['연도','구분'] )
print(df.head())

df = df.pivot_table(index=['연도','variable']  , columns=['구분']  , aggfunc='sum')

print()
df['검거율'] = df[('value','검거건수')] / df[('value','발생건수')]
#df['검거율'] = df['검거건수'] / df['발생건수']
# print(df.sort_values('검거율',ascending=False).reset_index())
# print(df.sort_values('검거율',ascending=False))
df = df.sort_values('검거율',ascending=False)
print(df)
print(df.groupby(['연도'])['검거율'].idxmax())
# df = pd.read_csv('crime.csv')

# df = pd.melt(df, id_vars=['연도','구분'])
# print(df.columns)
# cond = (df['구분'] == '검거건수' )& (df['variable'] == '강력범죄')

# print(df[cond].sum(numeric_only=True))

# print(4750)