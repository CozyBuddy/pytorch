import pandas as pd

menu = pd.Series(['비빔밥','김치찌개','된장찌개'])
print(menu)

price = pd.Series([10000,9000,8000])

print(price)

df = pd.DataFrame({
    '메뉴' : menu,
    '가격' : price
})

print(df)

df = pd.DataFrame({
    '메뉴' : ['비빔밥' , '김치찌개','된장찌개'] ,
    '가격' : [10000,9000,8000],
    '원산지' : ['국내산' , '국내산' , '중국산']
})

print(df)

print(df['메뉴'])

print(df[['메뉴']])

print(df[['메뉴','가격']])

print(type(df) , type(df['메뉴']), type(df[['메뉴']]))

df = pd.DataFrame({
    '메뉴' : ['아메리카노' , '카페라떼' , '카페모카' , '카푸치노' , '에스프레소' , '밀크티' , '녹차'],
    '가격' : [4500,5000,5500,5000,4000,5900,5300],
    '칼로리' : [10,110,250,110,20,210,10]
})

print(df)

df.to_csv('temp.csv')

temp_df = pd.read_csv('temp.csv')
print(temp_df)

df.to_csv('temp.csv', index=False)
df = pd.read_csv('temp.csv')
print(df.head())

print(df.sample(3))

print(df.shape)

print(df.info())

print(df.corr(numeric_only=True))

df_car = pd.DataFrame({
    'car' : ['Sedan' , 'SUV' , 'Sedan' , 'SUV' ,'SUV' ,'SUV' , 'Sedan' ,'Sedan' , 'Sedan' ,'Sedan' , 'Sedan'],
    'size' : ['S' ,'M' ,'S','S' ,'M' ,'M' ,'L' ,'S','S' ,'M','S']
})

print(df_car.head())

print(df_car.nunique())
print(df_car['car'].unique())
print(df_car['size'].unique())


print(df_car['car'].value_counts())
print(df_car['size'].value_counts())

print(df_car.value_counts())

print(df.describe())

print(df_car.describe(include='O'))


data = {'메뉴' : ['아메리카노' , '카페라떼' ,'카페모카','카푸치노' , '에스프레소' , '밀크티' ,'녹차'],
        '가격' : [4500.0,5000.0,5500.0,5000.0,4000.0,5900.0,5300.0],
        '칼로리' : ['10','110','250','110','20','210','0']}

df = pd.DataFrame(data)

print(df.info())

df['가격'] = df['가격'].astype('int')

print(df.info())

df['칼로리'] = df['칼로리'].astype('float')

print(df.info())

df = pd.read_csv('cafe.csv')
print(df.head())


df['newJeans'] = 0
print(df.head())

df.drop(1,axis=0,inplace=True)

print(df)

df = df.drop('칼로리',axis=1)
print(df.head())

df = pd.read_csv('cafe.csv')

print(df.loc[0])

print(df.loc[1,'가격'])

print(df.loc[:,'가격'])

print(df.loc[2,'메뉴':'칼로리'])

print(df.loc[2,['메뉴','칼로리']])

print(df.loc[1:3 , '메뉴':'가격'])

df = pd.read_csv('cafe.csv')

print(df.head())

df.drop(0,axis=0 , inplace=True)

print(df.head())

print(df.iloc[0])

print(df.iloc[:,1])

print(df.iloc[2,0:2])

df = pd.read_csv('cafe.csv')

print(df.head())

import numpy as np
df['원산지'] = np.nan
print(df.head())

df.loc[0,'원산지'] = '콜롬비아'

df.loc[2:3 , '원산지'] = '과테말라'

print(df.head())

df.loc['시즌'] = ['크리스마스라떼' , 6000,300,'한국']

print(df.head(10))

df.loc[7] = {'메뉴' : '딴짓커피' , '가격' : 2000 , '칼로리' : 20}

print(df.tail())

df.drop('시즌' , axis=0 , inplace=True)

df.to_csv('cafe2.csv' , index=False)


df = pd.read_csv('cafe2.csv')

print(df.sort_index(ascending=False))

print(df.sort_values('메뉴' , ascending=False))

df.sort_values(['가격','메뉴'] , ascending=[False,True] , inplace=True)
print(df.head())

df = df.reset_index(drop=True)

print(df.head())

df=pd.read_csv('cafe2.csv')

print(df.head())

print(df['칼로리'] <50)

cond = df['칼로리'] < 50
print(df[cond])

print(df[~cond])

cond1 = df['가격'] >=5000
cond2 = df['칼로리'] >100

print(df[cond1 & cond2])

print(df[cond1 | cond2])

cond = (df['원산지'] == '과테말라')
print(df[cond])

cond2 =df['가격'] <=5000
print(df[cond & cond2 ])

print(df['메뉴'].isin(['녹차']))

cond = df['메뉴'].isin(['녹차'])

print(df[cond])

box = ['녹차' , '카푸치노' , '카페라떼']

cond = df['메뉴'].isin(box)

print(df[cond])


df = pd.read_csv('cafe2.csv')

print(df.head())

print(df.isnull().sum())


df['원산지'] = df['원산지'].fillna('코스타리카')

print(df.head())

df.to_csv('cafe3.csv' ,index=False)


df = pd.read_csv('cafe3.csv')

print(df.head())
df.replace('아메리카노' , '룽고' , inplace=True)

df.replace('녹차' , '그린티' , inplace=True)

print(df.head(10))


change = {'룽고' : '아메리카노' , '그린티' : '녹차'}

df.replace(change,inplace=True)

print(df.head())

df.loc[6,'원산지'] = '대한민국'

print(df.head(7))

print(df.tail(3))

df.loc[1:2,'이벤트가'] = 1000
print(df.head())


df.to_csv('cafe4.csv' ,index=False)


import pandas as pd

df = pd.DataFrame({
    'A' : ['데이터 분석' , '기본 학습서' , '퇴근 후 열공'],
    'B' : [10,20,30],
    'C' : ['ab cd' , 'AB CD' , 'ab cd ']
})

print(df.head())

df['A'] = df['A'].replace('분석' , '시각화')

print(df.head())

df['A'] = df['A'].str.replace('분석' , '시각화')

print(df.head())

df['B'] = df['B'].replace(10,100)

print(df.head())

#df['B'] = df['B'].str.replace(20,200)

print(df['A'].str.split().str[1])

print(df['A'].str.split()[0][0])

df['D'] = df['A'].str.split().str[0]

print(df.head())

print(df['A'].str.contains('기본'))

cond = df['A'].str.contains('기본')

print(df[cond])

df['기본포함유무'] = df['A'].str.contains('기본')

print(df.head(10))


df['문자길이'] = df['A'].str.len()

print(df.head())


df['C'] = df['C'].str.lower()

print(df.head())

df['C'] = df['C'].str.upper()

print(df['C'] == 'AB CD')

df['C'] = df['C'].str.lower()
df['C'] = df['C'].str.replace(' ','')

print(df['C'])

print(df['C'].str[1:3])

print(df['C'][1:3])

df = pd.read_csv('cafe4.csv')

print(df.head())


cond = df['가격'] > 5000

print(sum(cond))

print(df.sum(numeric_only=True))

print(df.sum(axis=1 , numeric_only=True))

print(df['가격'].quantile(.25))

cond = df['가격'].quantile(.25) > df['가격']

print(df[cond])

print(df['원산지'].mode()[0])


print(df['가격'].max())

print(df.loc[df['가격'].idxmax()])

print(df.nlargest(3,'가격'))

print(df.sort_values('가격',ascending=False))


def cal(x):
    if x>=100:
        return 'No'
    elif x<100:
        return 'Yes'
    
df['먹어도 될까요?'] = df['칼로리'].apply(cal)

print(df.head())

df = pd.DataFrame({
    'Name' : {0 : '쿼카' , 1 :'알파카' , 2 : '시바견'},
    '수학' : {0:90 , 1:93 ,2:85} ,
    '영어' : {0:92 , 1:84 , 2:86} ,
    '국어' : {0:91 , 1:94, 2:83}
})

print(df.head())

print(pd.melt(df , id_vars=['Name']))


df = pd.DataFrame({
    '반' : {0:'A반' , 1:'A반' , 2:'B반'},
    '이름' : {0 : '쿼카' , 1 :'알파카' , 2 : '시바견'},
    '수학' : {0:90 , 1:93 ,2:85} ,
    '영어' : {0:92 , 1:84 , 2:86} ,
    '국어' : {0:91 , 1:94, 2:83}
})

print(df.head())

print(pd.melt(df , id_vars=['반','이름'] , var_name='과목' , value_name='점수'))

print(pd.melt(df , id_vars=['반','이름'] , var_name='과목' , value_name='점수'))

import pandas as pd

data = {
    'Date1' : ['2024-02-17' , '2024-02-18' , '2024-02-19' , '2024-02-20'],
    'Date2' : ['2024:02:17' , '2024:02:18' , '2024:02:19' , '2024:02:20'],
    'Date3' : ['24/02/17' , '24/02/18' , '24/02/19' , '24/02/20'],
    'Date4' : ['02/17/2024' , '02/18/2024', '02/19/2024','02/20/2024'],
    'Date5' : ['17-Feb-2024' , '18-Feb-2024','19-Feb-2024' ,'20-Feb-2024'],
    'DateTime1' : ['24-02-17 13:50:30' , '24-02-18 14:55:45' , '24-02-19 15:30:15' , '24-02-20 16:10:50'],
    'DateTime2' : ['2024-02-17 13-50-30' , '2024-02-18 14-55-45' , '2024-02-19 15-30-15' , '2024-02-20 16-10-50'],
    'DateTime3' : ['02/17/2024 01:50:30 PM' , '02/18/2024 02:55:45 PM' , '02/19/2024 03:30:15 AM' , '02/20/2024 04:10:50 AM' ],
    'DateTime4' : ['17 Feb 2024 13:50:30' , '18 Feb 2024 14:55:45' , '19 Feb 2024 15:30:15' , '20 Feb 2024 16:10:50']
}

df = pd.DataFrame(data)

df.to_csv('date_data.csv' ,index=False)

print(df)

print(df.info())

df = pd.read_csv('date_data.csv')
df['Date1'] = pd.to_datetime(df['Date1'])
df['Date2'] = pd.to_datetime(df['Date2'] ,format='%Y:%m:%d')
df['Date3'] = pd.to_datetime(df['Date3'] ,format='%y/%m/%d')
df['Date4'] = pd.to_datetime(df['Date4']) 
df['Date5'] = pd.to_datetime(df['Date5'])

df['DateTime1'] = pd.to_datetime(df['DateTime1'] , format='%y-%m-%d %H:%M:%S')
df['DateTime2'] = pd.to_datetime(df['DateTime2'] , format='%Y-%m-%d %H-%M-%S')
df['DateTime3'] = pd.to_datetime(df['DateTime3'] )
df['DateTime4'] = pd.to_datetime(df['DateTime4'])

print(df)
print(df.info())


df = pd.read_csv('date_data.csv')
df['Date2'] = pd.to_datetime(df['Date2'] , format='%Y:%m:%d')
df['Date3'] = pd.to_datetime(df['Date3'] , format='%y/%m/%d')
df['DateTime1'] = pd.to_datetime(df['DateTime1'] , format='%y-%m-%d %H:%M:%S')
df['DateTime2'] = pd.to_datetime(df['DateTime2'] , format='%Y-%m-%d %H-%M-%S')

print(df[['Date2' , 'Date3' , 'DateTime1' , 'DateTime2']])

df['year'] = df['DateTime1'].dt.year
df['month'] = df['DateTime1'].dt.month
df['day'] = df['DateTime1'].dt.day
df['hour'] = df['DateTime1'].dt.hour
df['minute'] = df['DateTime1'].dt.minute
df['second'] = df['DateTime1'].dt.second

print(df)

df['dayofweek'] = df['DateTime1'].dt.dayofweek

print(df)


df = pd.read_csv('date_data.csv' , usecols=['DateTime4'] , parse_dates=['DateTime4'])

print(df)

day = pd.Timedelta(days=99)
df['100days'] = df['DateTime4']+ day

print(df)

hour = pd.Timedelta(hours=100)

df['100hours'] = df['DateTime4'] + hour

print(df)

diff = pd.Timedelta(weeks=7 , days=7 , hours=7 , minutes=7 , seconds=7)
df['diff'] = df['DateTime4'] - diff

print(df)

diff = df['100hours'] - df['diff']

print(diff)

print(diff.dt.total_seconds())

print(diff.dt.days)

min = 5.41

print(int(min) , '분')
print(0.41*60 , '초')

print(round(diff.dt.total_seconds()/60))

import pandas as pd

appetizer = pd.DataFrame({
    'Menu' : ['Salad' , 'Soup' , 'Bread'] ,
    'Price' : [5000,3000,2000]
})

main = pd.DataFrame({
    'Menu' : ['Steak' , 'Pasta' , 'Chicken'] ,
    'Price' : [15000,12000,10000]
})

print(appetizer)
print(main)

full_menu = pd.concat([appetizer , main] , ignore_index=True)

print(full_menu)

full_menu = pd.concat([appetizer, main] , axis=1)
print(full_menu)

import pandas as pd

data = { '이름' : ['서아' , '다인' , '채아' , '예담' , '종현' , '태헌'],
        '부서' : ['개발' , '기획' , '개발' , '기획' , '개발' , '기획'],
        '급여' : [3000,3200,3100,3300,2900,3100]}

df = pd.DataFrame(data)

print(df)

pt = df.pivot_table(index='부서' , values='급여',aggfunc='mean')
print(pt)

import pandas as pd
import numpy as np

df = pd.DataFrame({'메뉴' : ['아메리카노' , '카페라떼' , '에스프레소' , '카페모카' , '바닐라라떼'] ,
                   '가격' : [4500,5000,4000,5900,5300] ,
                   '칼로리' : [10,110, np.NaN , 210 , np.NaN] ,
                   '원두' : ['과테말라' , '브라질' , '과테말라' , np.NaN , np.NaN]})

print(df)

#Q1
min = df['칼로리'].min()
df['칼로리'] = df['칼로리'].fillna(min)
print(df)

#Q2

medi = df['원두'].mode()[0]

df['원두'] = df['원두'].fillna(medi)

print(df)

#Q3

cond = df['가격'] > 5000

print(df[cond])

#Q4

df['이벤트가'] = df['가격'] *0.5
print(df)

#Q5
df.drop(['칼로리'] , axis=1 , inplace=True)
print(df)

#Q6
print(df.loc[0:2,:])

#Q7
print(df.iloc[0:3 , :])

#Q8
print(df.loc[1:2 , ['메뉴' , '가격']])

#Q9

print(df.iloc[1:3 , 0:2 ])

#Q10
df = df.sort_values(['가격'] , ascending=False)
print(df.head(3))
