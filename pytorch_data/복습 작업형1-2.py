import pandas as pd

df = pd.read_csv('delivery_time.csv')

df['주문시간'] = pd.to_datetime(df['주문시간'])

df['주문시간'] = df['주문시간'].dt.to_period('M')

#1
print(df['주문시간'].value_counts().index[0])

#2

selected = df['주문시간'].value_counts().index[0]

cond = df['앱종류'] == '배고팡'

cond2 = df['결제종류'] == '앱결제'

df = df[cond]
print(round(cond2.mean(),2))


#1

df = pd.read_csv('delivery_time.csv')

df['주문시간'] = pd.to_datetime(df['주문시간'])

cond = (df['주문시간'].dt.hour >=10) & (df['주문시간'].dt.hour <13)
print(df[cond])

#2
df = df[cond]
df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['배달시간'] = (df['실제도착시간'] -  df['주문시간']).dt.total_seconds() /3600

df['속도'] = df['거리'] / df['배달시간']

print(df.head())

cond = df['속도'] >=50

print(len(df[cond]))

#1

df = pd.read_csv('delivery_time.csv')

df['주문시간'] =  pd.to_datetime(df['주문시간'])

df['주문시간'] = df['주문시간'].dt.to_period('M')
print(df.head())

print(df.groupby('주문시간').size())

#2

df = df.groupby('주문시간').size()

print(str(df.sort_values(ascending=False).index[0]).replace('-',''))

#1
df = pd.read_csv('delivery_time.csv')

def calculate(e):
    if(e<5):
        return 2000
    elif(e>=5 and e<10):
        return 4000
    elif(10<=e and e<15):
        return 6000
    else:
        return 8000
    
df['배달료'] = df['거리'].apply(calculate)

print(df.head())

#2
df['주문시간'] = pd.to_datetime(df['주문시간'])
df['월'] = df['주문시간'].dt.month

print(df.head())

print(df.groupby('월').sum(numeric_only=True))

#3

df = df.groupby('월').sum(numeric_only=True)

print(df.sort_values('배달료',ascending=False).index[0])
print(int(df.sort_values('배달료',ascending=False).iloc[0][1]))

#1
df = pd.read_csv('delivery_time.csv')

df['주문시간'] = pd.to_datetime(df['주문시간'])
df['평일주말'] = df['주문시간'].dt.dayofweek
print(df)

cond =( df['평일주말'] ==5 ) | (df['평일주말'] ==6 )

print(len(df[~cond]))
print(len(df[cond]))


#2

print(int(abs(len(df[~cond]) - len(df[cond]))))

#1

df = pd.read_csv('delivery_time.csv')

print(df.head())

df['usersplit'] = df['user'].str.replace('user_','').astype('int')

print(df.head())

print(df['usersplit'].sum())


#1
df = pd.read_csv('school_data.csv')

print(df.head())

df['합계'] = df['수학'] + df['국어'] + df['영어']

print(df.head())

#2

print(df.sort_values('합계' , ascending=False).iloc[:10])

#3
df = df.sort_values('합계' , ascending=False).iloc[:10]
print(int(round(df['수학'].mean(),0)))

#1

df = pd.read_csv('school_data.csv')

print(df.melt(id_vars=['이름'] , value_vars=['수학','영어','국어']))

df = df.melt(id_vars=['이름'] , value_vars=['수학','영어','국어'])

print(df['value'].sort_values(ascending=True).iloc[:25])

df = df['value'].sort_values(ascending=True).iloc[:25]

print(df.sum())


#1
df = pd.read_csv('school_data.csv')

df_science = pd.read_csv('school_data_science.csv')


df2 = pd.concat([df,df_science] , axis=1)

print(df2)

df2['평균'] = (df2['국어'] + df2['수학'] + df2['영어'] + df2['과학']) /4
print(df2.head())

#3

cond = df2['평균'] >=60

print(len(df2[cond]))

#1

df = pd.read_csv('school_data.csv')

df_social = pd.read_csv('school_data_social.csv')

Mdf = pd.merge(df,df_social , on='이름')

print(Mdf.head())

#2

cond = (Mdf['영어교사'] =='장선생') & (Mdf['사회교사'] =='오선생')

print(Mdf[cond])

#3
df = Mdf[cond]

print(int(df['수학'].sum()))

#1

df = pd.read_csv('sales.csv')

print(df)

print(df.groupby('지역코드')['판매금액'].transform('mean'))
# means = df['판매매']
# df['판매금액'].fillna(method='mean')

df['판매금액'] = df['판매금액'].fillna(df.groupby('지역코드')['판매금액'].transform('mean'))
df['평균금액'] = df.groupby('지역코드')['판매금액'].transform('mean')
print(df)

#2

df['절대차이'] = abs(df['판매금액'] -df['평균금액'])

print(df)

#3

print(df.groupby('지역코드').mean(numeric_only=True).sort_values('절대차이' , ascending=False).index[0])


df = pd.read_csv('store_sales.csv')
print(df)

df['매출액'] = df['판매수량'] * df['단가']

print(df)

#2

def setweek(e):
    if(e=='월' or e=='화'  or e=='수'  or e=='목'  or  e=='금' ):
        return '평일'
    else:
        return '주말'
df['주말여부'] = df['요일'].apply(setweek)

print(df.groupby(['매장코드','주말여부'])['매출액'].sum(numeric_only=True).unstack())

#3
df = df.groupby(['매장코드','주말여부'])['매출액'].sum(numeric_only=True).unstack()
df['차이'] = abs(df['주말'] - df['평일'])

print(df.sort_values('차이',ascending=False).iloc[0,2])

#1

df = pd.read_csv('region_sales.csv')

print(df)

ptable = pd.pivot_table(df , index=['Region','Channel'] , columns='Product' , values='Sales' , aggfunc='sum')

print(ptable)

ptable['총매출'] = ptable['A'] + ptable['B']
ptable['A비율'] = ptable['A'] / ptable['총매출']


print(round(ptable.sort_values('A비율',ascending=False).iloc[0][3],2))


#1
df = pd.read_csv('monthly_sales.csv')

df = pd.melt(df, value_vars=['Jan','Feb','Mar'] , id_vars=['Region'])
print(df.groupby(['Region','variable']).sum(numeric_only=True))

df = df.groupby(['Region','variable']).sum(numeric_only=True)

#2

cond = df['value'] > 1400

print(len(df[cond]))