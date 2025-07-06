import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_rows',20)
#22
df = pd.read_csv('delivery_time.csv')

print(df)

#1

print(df.info())

df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['주문시간'] = pd.to_datetime(df['주문시간'])
df['diff']= (df['실제도착시간'] - df['주문시간']).dt.total_seconds()/60

print(df)

#2

df= df.groupby(['앱종류']).mean(numeric_only=True)

print(df)


#3

minv = df['diff'].min()
print(int(round(minv,0)))


#23

#1
df = pd.read_csv('delivery_time.csv')

#1
print(df)


df['실제도착시간'] = pd.to_datetime(df['실제도착시간'])
df['예상도착시간'] = pd.to_datetime(df['예상도착시간'])

df['지연여부'] = (df['실제도착시간'] - df['예상도착시간']).dt.total_seconds() > 0 


print(df)
df = df.groupby('결제종류')['지연여부'].mean() # True False 컬럼으로 평균을 내면 그게 비율이 됨 ㅋㅋㅋㅋㅋㅋㅋ
print(round(df.max(),2))


#24

#1
df = pd.read_csv('delivery_time.csv')

print(df)
df2 = df.groupby('user').sum(numeric_only=True) # user가 인덱스가 됨
cond = df2['거리'] >=50
df2 = df2[cond]
print(df2.index)


#cond2 = df['user'] ==  df2['user']

filtered_data = df[df['user'].isin(df2.index)]

print(filtered_data)

paymethod = filtered_data['결제종류'].value_counts()
print(paymethod[0])
#print(len(filtered_data[filtered_data['결제종류'] == paymethod]))


#25

#1

df = pd.read_csv('delivery_time.csv')

print(df)

df['주문시간'] = pd.to_datetime(df['주문시간'])

df2 = (df.groupby('user')['주문시간'].max() - df.groupby('user')['주문시간'].min()).dt.days
#df3 = df.groupby('user')['주문시간'].min()
print(df2)
#2

cond = df2 >0

print(df2[cond].mean())

#3
cond2 = df2 > df2[cond].mean()

print(len(df2[cond2]))

#26

#1
df = pd.read_csv('delivery_time.csv')
df['주문확인'] = True
df['주문시간'] = pd.to_datetime(df['주문시간']).dt.to_period('M') 
df2 = df.groupby('주문시간')['주문확인'].sum()
print(df2.sort_values(ascending=False).reset_index().loc[0,'주문확인'])
target = df2.sort_values(ascending=False).index[0]
target2 = df2.sort_values(ascending=False).reset_index().loc[0,'주문확인']
print(target)
#2
print(df)
cond = (df['앱종류'] == '배고팡') & ( df['결제종류'] =='앱결제') & (df['주문시간'] == target)
cond2 = (df['앱종류'] == '배고팡') & (df['주문시간'] == target)
print(round(len(df[cond])/len(df[cond2]),2)) 


#27

#1
df = pd.read_csv('delivery_time.csv')

df['주문시'] = pd.to_datetime(df['주문시간']).dt.hour
df['주문시간'] =  pd.to_datetime(df['주문시간'])
df['실제도착시간'] =  pd.to_datetime(df['실제도착시간'])
cond =( df['주문시'] >=10 ) & (df['주문시'] <=13)
print(df[cond])
df= df[cond]
#2
df['과속여부'] = df['거리']/((df['실제도착시간'] - df['주문시간']).dt.total_seconds()/3600) >=50
cond = df['과속여부'] == True
print(df[cond])
print(len(df[cond]))

#28

#1
df = pd.read_csv('delivery_time.csv')

df['주문시간'] = pd.to_datetime(df['주문시간']).dt.to_period('M')

print(df.groupby('주문시간').size())

#2

df = df.groupby('주문시간').size()

cond = df.sort_values(ascending= False)

print(str(cond.reset_index().loc[0,'주문시간']).replace('-','')) 


#29

#1

df = pd.read_csv('delivery_time.csv')

print(df)


#df['배달료'] =  if df['거리'] < 5 
def caculate(e):
    if e < 5 :
     return 2000
    elif e <10 :
     return 4000
    elif e <15 :
     return 6000
    else :
     return 8000
    
df['배달료'] = df['거리'].apply(caculate)

print(df)


#2
df['월'] = pd.to_datetime(df['주문시간']).dt.month

print(df.groupby('월')['배달료'].sum())

#3
df = df.groupby('월')['배달료'].sum()
print(int(df.sort_values(ascending=False).reset_index().loc[0,'배달료']))


#30

#1

df = pd.read_csv('delivery_time.csv')

df['주말여부'] = pd.to_datetime(df['주문시간']).dt.dayofweek 

cond =( df['주말여부'] == 5) | (df['주말여부'] == 6)

print(len(df[cond]))
print(len(df[~cond]))

#2

print(abs(len(df[cond]) - len(df[~cond])))


#31

df = pd.read_csv('delivery_time.csv')

df['user'] = df['user'].str.replace('user_','').astype(int)

print(df)

count = df['user'].sum()
print(count)

pd.set_option('display.max_rows',20)
#32

#1
df = pd.read_csv('school_data.csv')

print(df)

df['총합'] = df['수학']+df['영어']+df['국어']

print(df)

#2
print(df.sort_values('총합',ascending=False).reset_index().loc[0:9])


#3
df = df.sort_values('총합',ascending=False).reset_index().loc[0:9]

print(round(df['수학'].mean()))


#33

#1

df = pd.read_csv('school_data.csv')

melted_df = df.melt(id_vars=['이름'] , value_vars=['수학','영어','국어'])
print(melted_df)

print(melted_df.sort_values('value', ascending=True).reset_index().loc[0:24])

print(melted_df.sort_values('value', ascending=True).reset_index().loc[0:24]['value'].sum())

#34

#1

df = pd.read_csv('school_data.csv')
df_science = pd.read_csv('school_data_science.csv')

df = pd.concat([df,df_science], axis=1)
print(df)

#2

df['평균'] = (df['수학'] + df['영어'] + df['국어'] + df['과학']) / 4

print(df)

cond = df['평균'] >=60

print(len(df[cond]))

#35

#1
df = pd.read_csv('school_data.csv')

df_social = pd.read_csv('school_data_social.csv')

df = pd.merge(df,df_social , on='이름')

print(df)

#2
cond = (df['영어교사'] == '장선생') & (df['사회교사'] == '오선생')

print(df[cond])

#3

df = df[cond]

print(int(df['수학'].sum()))


#36

#1

df = pd.read_csv('sales.csv')

#print(df)
df2 = df.groupby('지역코드')['판매금액'].transform('mean')

print(df2)

df['판매금액'].fillna(df2 , inplace=True)

print(df)

#2

df['차이'] = abs(df['판매금액'] - df2)

print(df)

#3

df3 = df.groupby('지역코드')['차이'].mean()

print(df3.sort_values(ascending=False).reset_index().loc[0,'지역코드'])


#37

#1

df = pd.read_csv('store_sales.csv')

print(df)

df['매출액'] = df['판매수량'] * df['단가']

print(df)

#2

df['주말여부'] = (df['요일'] == '토') | (df['요일'] =='일')

print(df)

df2 = df.groupby(['매장코드','주말여부'])['매출액'].sum().unstack()

print(df2)


#3
df2['주말평일차이'] = abs(df2[False] - df2[True])

print(df2.sort_values('주말평일차이',ascending=False).reset_index().loc[0,'주말평일차이'])


#38

#1
df = pd.read_csv('region_sales.csv')

print(df)

pivot = pd.pivot_table(df , index=['Region','Channel'] , columns='Product' , values='Sales' , aggfunc='sum')

print(pivot)


#2

pivot['총매출'] = pivot['A'] + pivot['B']

pivot['A비율'] = pivot['A']/pivot['총매출']

print(pivot)
#3

result = pivot['A비율'].max()

print(round(result,2))

#39

#1
df = pd.read_csv('monthly_sales.csv')

#1

print(df)

df = pd.melt(df , id_vars='Region' , value_vars=['Jan','Feb','Mar'] , var_name='Month' , value_name='Sales')

print(df)

#2

groupsales = df.groupby(['Region','Month'])['Sales'].sum().reset_index()
print(groupsales)
cond = groupsales['Sales'] > 1400

print(len(groupsales[cond]))