import pandas as pd

#1
df = pd.read_csv('type1_data2.csv' , index_col='year')

print(df.loc[2001])

cond = df.loc[2001] > df.loc[2001].mean()

print(len(df.loc[2001][cond]))

#2

m2 = df.loc[2003].mean()
print(m2)
print(sum(df.loc[2003] < m2))

#3

print(sum(df.loc[2003] < m2) + len(df.loc[2001][cond]))


#1

df = pd.read_csv('type1_data1.csv')

df = df.fillna(method='bfill')

print(df)

#2
df = df.groupby(['city','f2']).sum(numeric_only=True).reset_index()

print(df)

#3
print('asdffffffffffffff')
print(df.sort_values('views' ,ascending=False).iloc[2,0])

df = pd.read_csv('type1_data1.csv')

#1

print(df)
df['subscribed'] = pd.to_datetime(df['subscribed'])

df['subscribed'] = df['subscribed'].dt.month
print(df.head())

print(df.groupby('subscribed').sum(numeric_only=True))

#2
df = df.groupby('subscribed').sum(numeric_only=True)

print(df.sort_values('views',ascending=True).index[0])


#1

df = pd.read_csv('delivery_time.csv')
print(df.info())
print(df)

df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['예상도착시간'] = pd.to_datetime(df['예상도착시간'])

cond = (df['실제도착시간'] - df['예상도착시간']).dt.total_seconds() > 0
print(len(df[cond]))

df = df[cond]

cond = df['거리'] >= 7

print(len(df[cond]))

df = pd.read_csv('delivery_time.csv')

print(df.head())

#1
df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['주문시간'] = pd.to_datetime(df['주문시간'])
df['분차이'] = (df['실제도착시간'] - df['주문시간']).dt.total_seconds() / 60

#2
print(df.groupby('앱종류')['분차이'].mean())

#3
df = df.groupby('앱종류')['분차이'].mean()
print(round(df.sort_values().iloc[0]))

#1

df = pd.read_csv('delivery_time.csv')

df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['예상도착시간'] = pd.to_datetime(df['예상도착시간'])


df['늦음여부'] = (df['실제도착시간'] - df['예상도착시간']  ).dt.total_seconds() > 0 


print(df.groupby('앱종류')['늦음여부'].mean())

#2
df = df.groupby('결제종류')['늦음여부'].mean()

print(round(df.sort_values(ascending=False).iloc[0],2))

#1
df = pd.read_csv('delivery_time.csv')

cond = df['거리'] >= 50


print(df.groupby('user')['거리'].sum())

df2 = df.groupby('user')['거리'].sum()


cond =  df2 > 50
#print(df2[cond])

print(df[df['user'].isin(df2[cond].index)][['user','결제종류' ]])

print(df[df['user'].isin(df2[cond].index)]['결제종류'].value_counts())




df = pd.read_csv('delivery_time.csv')

#1

df['주문시간'] = pd.to_datetime(df['주문시간'])

print((df.groupby('user')['주문시간'].max() -df.groupby('user')['주문시간'].min()).dt.days)

df = (df.groupby('user')['주문시간'].max() -df.groupby('user')['주문시간'].min()).dt.days

print(df[df != 0])
#3
df = df[df!=0]
cond = df.mean()

print(cond)

cond = df > cond

print(len(df[cond]))

