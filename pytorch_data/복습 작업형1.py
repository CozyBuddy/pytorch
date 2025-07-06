import pandas as pd

df = pd.read_csv('type1_data1.csv')

print(df)

#1
cond = df['f5'] != 0
print(df[cond])
df = df[cond]
#2
min = df['views'].min()

df['views'].fillna(min , inplace=True)

print(df.head(50))

#3
median = df['views'].median()

print(int(median))


#1

df = pd.read_csv('type1_data1.csv')

date= df['subscribed'].mode()[0]

print(date)

#2

date= date.split('-')[2]

print(int(date))


#그외

df = df['subscribed'].value_counts()

print(df)
#1
df = pd.read_csv('type1_data1.csv')

df = df.dropna()

print(df)

#2

df['new'] = df['views'] / df['f1']

print(df)

#3

max = df['new'].max()

cond = df['new'] == max

df = df.sort_values('new' , ascending=False)

print(int(df.iloc[0,1]))


df = pd.read_csv('type1_data1.csv')

#1

df['views'].fillna(0 , inplace=True)

#2

df = df.sort_values('views', ascending=False)

print(df.iloc[:10,9])


#3
print( df.iloc[9,9])
df.iloc[:10,9] = df.iloc[9,9]

print(df)

#4

allsum = df['views'].sum()
print(int(allsum))


df = pd.read_csv('type1_data1.csv')

print(df)
#1
cond = df['f4'].str.contains('FJ')
print(df[cond]) 

#2
mean = df[cond]['f2'].mean()

print(round(mean , 2))

df = pd.read_csv('type1_data1.csv')

#1
cond = (df['f3'] == 'gold') & (df['f2'] == 2)

print(df[cond])


#2
vars = df[cond]['f1'].var()

print(round(vars ,2))


#1

df = pd.read_csv('type1_data1.csv')

df['age'] = df['age'] +1
print(df)

#2

cond = (df['age'] >=20) & (df['age'] < 30 )
cond2 = ( df['age'] >=30) & (df['age'] <40)

mean1 = df[cond]['views'].mean()

mean2 = df[cond2]['views'].mean()

print(round(abs(mean1 - mean2),2))

df = pd.read_csv('type1_data1.csv')

print(df)

#1
cond = df['subscribed'].str.contains('2024-02')

print(df[cond])

#2

cond2 = df['f3'] == 'gold'
print(len(df[cond & cond2]['f3']))


df = pd.read_csv('type1_data1.csv')


#1

cond = df['views'] <= 1000
print(df[cond])

#2

print(df[cond]['f4'].mode()[0])


#1

df = pd.read_csv('type1_data1.csv')

df = df.dropna()

#2

means = df.groupby('city').mean(numeric_only=True)

print(means)

#3
means = means.sort_values('f2',ascending=False)

print(means.index[0])

df= pd.read_csv('type1_data1.csv')

#1

df = df.dropna()

#2

data7 = int(len(df)*0.7)

print(df.iloc[:data7 ,:])

#3
df = df.iloc[:data7 ,:]

int1 = df['views'].quantile(0.75)
int2 = df['views'].quantile(0.25)

print(int(int1 - int2))

df = pd.read_csv('type1_data1.csv')

#1

cond = df.isna()

print(df.isna().sum())

print(df.isna().sum().sort_values( ascending=False).iloc[:2])

#2

df = df.dropna(subset=['f1'])

print(df)

#3
modes = df['f3'].mode()[0]

df['f3'].fillna(modes, inplace=True)


print(df)

#4

cond = df['f3'] == 'gold'

print(len(df[cond]))

#1

df = pd.read_csv('type1_data1.csv')

cond = df['f1'].isna()

print(df[cond])

#2

means = df[cond]['age'].mean()

print(round(means,1))

df = pd.read_csv('type1_data1.csv')

#1
df = df.drop_duplicates()

print(df)

#2

df['f3'] = df['f3'].fillna(0)

df['f3'] = df['f3'].replace('silver' ,1).replace('gold',2).replace('vip',3)


#3
print(df['f3'].sum())


#1

df = pd.read_csv('type1_data1.csv')

print(df.select_dtypes(include='O').columns)
df = df.drop(df.select_dtypes(include='O').columns , axis=1)

print(df)

#2

df.fillna(0 ,inplace=True)

#3

cond = df['age'] + df['f1'] + df['f2'] + df['f5']+ df['views'] > 3000


print(len(df[cond]))

df = df.T

print(sum(df.sum() > 3000))

df = pd.read_csv('type1_data1.csv')

#1

print(df['views'].quantile(0.25))
print(df['views'].quantile(0.75))
print(df['views'].quantile(0.75) - df['views'].quantile(0.25) )
a = df['views'].quantile(0.25)
b = df['views'].quantile(0.75)
#2
cond = ( df['views'] < a - (b-a)*1.5 ) | ( df['views'] > b + (b-a)*1.5)

print(df[cond])

#3

print(int(df[cond]['views'].sum()))

#1

df = pd.read_csv('type1_data1.csv')

print(df['views'].var() ** 0.5)
std1 = df['views'].std()
#2

cond = df['views'].dtype

cond = df['age'] % 1 == 0

cond2 = df['age'] >= 1


print(df[(cond | ~cond2 )])

#3
df = df[(cond & cond2 )]
std2 = df['views'].std()

print(round(std1+std2 ,2))

