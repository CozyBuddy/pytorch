import pandas as pd

pd.set_option('display.max_rows', None)
df = pd.read_csv('type1_data1.csv')

print(df)

cond = df['f5'] != 0

#1
#df[cond]

print(df[cond])

#2
df = df[cond]
minv = df['views'].min()

df['views'] = df['views'].fillna(minv)

print(df)

#3
print(int(df['views'].median()))


df = pd.read_csv('type1_data1.csv')

#22

#1
modevalue = df['subscribed'].mode()[0]

print(modevalue)

#2

print(int(modevalue.split('-')[2]))

#33

#1
df = pd.read_csv('type1_data1.csv')

df = df.dropna()

print(df)

#2
df['새로운'] = df['views'] / df['f1']

print(df)

#3

maxv = df['새로운'].max()

cond = df['새로운'] == maxv
print(df[cond])

print(int(df[cond].reset_index().loc[0,'age']))

#44

#1
#df = 

df = pd.read_csv('type1_data1.csv')

print(df['views'].fillna(0 , inplace=True))
print(df)

#2

df = df.sort_values('views' , ascending=False)

print(df.reset_index().loc[9:9,'views'])

#3
replacevalue = int(df.reset_index().loc[ 9 , 'views'])
print(replacevalue)

df= df.reset_index()
df.loc[0:9, 'views'] = replacevalue

print(df) 

#4
sums = df['views'].sum()
print(int(sums))


#55

#1
df = pd.read_csv('type1_data1.csv')

print(df)

cond = df['f4'].str.contains('FJ')

print(df[cond])

#2
df = df[cond]

meanv = df['f2'].mean()
print(round(meanv,2))

#66

#1
df = pd.read_csv('type1_data1.csv')

cond = df['f3'] == 'gold'

cond2 = df['f2'] == 2

print(df[cond & cond2])


#2

df = df[cond & cond2]

varv = df['f1'].var()

print(round(varv,2))


##7

#1
df = pd.read_csv('type1_data1.csv')


df['age'] = df['age']+1

print(df)

#2

cond1 = (df['age'] >=20) & (df['age'] < 30)

cond2 = (df['age'] >=30) &( df['age'] <40)

df1 = df[cond1]['views'].mean()
df2 = df[cond2]['views'].mean()
print(round(abs(df1-df2),2))


#88
#1

df = pd.read_csv('type1_data1.csv')

print(df['subscribed'])

cond = df['subscribed'].str.contains('2024-02')
df['subscribed'] = pd.to_datetime(df['subscribed'])

cond3 = df['subscribed'].dt.year == 2024
cond4 = df['subscribed'].dt.month == 2

print(df[cond3 & cond4])
print(df[cond])

#2

cond2 = df['f3'] == 'gold'

print(len(df[cond2 & cond]))


#99

#1
df = pd.read_csv('type1_data1.csv')

df['views'].dropna(inplace = True) 

cond = df['views'] <=1000

print(df[cond])

#2
df = df[cond]
cond2 = df['f4'].mode()[0]
print(cond2)


#10

#1
df = pd.read_csv('type1_data1.csv')

df = df.dropna()

print(df)

#2

df = df.groupby(['city']).mean(numeric_only=True)

print(df)

#3

cond = df['f2'] == df['f2'].max()
print(df[cond].reset_index().loc[0,'city'])


#11

#1
df = pd.read_csv('type1_data1.csv')

df.dropna(inplace=True)

print(df)

#2

cond = int(len(df)*0.7)

print(df.iloc[:cond])
df = df.reset_index().loc[:cond-1]

print(df)

#3

qv = int(df['views'].quantile(0.75) - df['views'].quantile(0.25))
print(qv)


#12

#1
df = pd.read_csv('type1_data1.csv')

print(df.isna().sum())


#2
df = df.dropna(subset=['f1'])

print(df)

#3

modev = df['f3'].mode()[0]
df['f3'] = df['f3'].fillna(modev)

print(df)


#4

cond = df['f3'] == 'gold'

print(int(len(df[cond])))



#13

df = pd.read_csv('type1_data1.csv')

cond = df['f1'].isna() == True

print(df[cond])

df = df[cond]

#2

meanv = round(df['age'].mean() ,1 )

print(meanv)


#14

df = pd.read_csv('type1_data1.csv')

#1

df = df.drop_duplicates()

print(df)

#2

df['f3'].fillna(0, inplace=True)

df['f3'] = df['f3'].replace('silver',1).replace('gold',2).replace('vip',3)
print(df)

#3
print(int(df['f3'].sum()))

#15

#1

df = pd.read_csv('type1_data1.csv')

print(df.info())
print(df.select_dtypes(exclude='object').columns)
df.drop(columns=['id','city','f3','f4','subscribed'] , inplace=True)
#print(df.dropna(subset=['id','city','f3','f4','subscribed'] , inplace=True))
print(df)

#2
df.fillna(0 ,inplace=True)

print(df)

#3
cond = df['age'] + df['f1']+df['f2'] +df['f5'] +df['views'] >3000

print(len(df[cond]))

#16

#1
df = pd.read_csv('type1_data1.csv')

first = df['views'].quantile(0.25)
second = df['views'].quantile(0.75)

iqr = second -first

print(first,second,iqr)

#2
cond = (df['views'] <= first- iqr*1.5) | (df['views'] > second+iqr*1.5)

print(df[cond])

#3
df= df[cond]

print(int(df['views'].sum()))

#18

df = pd.read_csv('type1_data2.csv', index_col='year')

print(df)

#1
m1 = df.loc[2001].mean()
cond1 = df.loc[2001] > m1


print(len(df.loc[2001][cond1]))

#2
m2 = df.loc[2003].mean()
cond2 = df.loc[2003] < m2
print(len(df.loc[2003][cond2]))

#3
result = len(df.loc[2001][cond1]) + len(df.loc[2003][cond2])

print(result)


#19

df = pd.read_csv('type1_data1.csv')

#1
df = df.fillna(method='bfill')

print(df)

#2

df = df.groupby(['city','f2']).sum(numeric_only=True)
print(df)

#3
df = df.sort_values('views', ascending=False)

print(df.reset_index().loc[2]['city'])


#20
#1
df = pd.read_csv('type1_data1.csv')

df['subscribed'] = pd.to_datetime(df['subscribed']).dt.month

print(df.info())

df = df.groupby('subscribed').sum(numeric_only=True)

print(df)

#2
df = df.sort_values('views', ascending=True)
print(int(df.reset_index().loc[0]['subscribed']))


#21

#1
df = pd.read_csv('delivery_time.csv')

#print(df)

df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['예상도착시간'] = pd.to_datetime(df['예상도착시간'])

cond = df['실제도착시간'] > df['예상도착시간']
cond3 = (df['실제도착시간'] - df['예상도착시간']).dt.total_seconds() > 0
print(df[cond3])
print(len(df[cond]))


#2
df = df[cond3]

cond2 = df['거리'] >= 7

print(len(df[cond2]))


