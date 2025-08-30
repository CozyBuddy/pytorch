import pandas as pd

menu = pd.Series(['비빔밥','김치찌개','된장찌개'])
print(menu)

price = pd.Series([10000,9000,8000])
print(price)


df = pd.DataFrame({
    "메뉴": menu ,
    "가격": price
})

print(df)
print(df['메뉴'])
# df['메뉴'] => 시리즈 형태 , df[['메뉴']] => 데이터 프레임 형태
print(df[['메뉴']])

print(df[['메뉴','가격']])

cols = ['메뉴','가격']
print(df[cols])

print("df   " , type(df))
print("df['가격']" , type(df['가격']))
print("df[['가격']]" , type(df[['가격']]))


df = pd.DataFrame({
    "메뉴" : ['아메리카노','카페라떼' , '카페모카' , '카푸치노' , '에스프레소','밀크티','녹차'] ,
    "가격" : [4500,5000,5500,5000,4000,5900,5300] ,
    "칼로리" : [10,110,250,110,20,210,0]
})

print(df)

df.to_csv('temp.csv')

temp_df = pd.read_csv('temp.csv')
print(temp_df.head())

df.to_csv('cafe.csv', index=False)
df = pd.read_csv('cafe.csv')
print(df.head())

print(pd.read_csv('cafe.csv' , index_col=0))

print(pd.read_csv('cafe.csv', usecols=['가격']))

# 데이터 샘플 확인
print(df.head(2))
print(df.tail(3))

print(df.sample(3))
print(df.shape)

print(df.info())

print(df.corr(numeric_only=True))

df_car = pd.DataFrame({
    "car" : ['Sedan','SUV','Sedan','SUV','SUV','SUV','Sedan','Sedan','Sedan','Sedan','Sedan'],
    "size" : ['S','M','S','S','M','M','L','S','S','M','S']
})

print(df_car.head(3))

# 컬럼별로 고유한 값의 개수
print(df_car.nunique())

#고유한 값의 구체적 항목
print(df_car['car'].unique())
print(df_car['size'].unique())

# 한번에 파악

print(df_car['car'].value_counts())
print(df_car['size'].value_counts())

print(df.describe())

print(df_car.describe())
print(df_car.describe(include='O'))

df = pd.read_csv('cafe.csv')
print(df.describe(include='O'))

data = {
    '메뉴' : ['아메리카노','카페라떼','카페모카','카푸치노','에스프레소','밀크티','녹차'],
    "가격" : [4500.0,5000.0 , 5500.0,5000.0,4000.0,5900.0,5300.0],
    "칼로리" : ['10','110','250','110','20','210','0']
}

df = pd.DataFrame(data)
print(df.info())

df['가격'] = df['가격'].astype('int64')
print(df.info())

df['칼로리'] = df['칼로리'].astype('float')
print(df.info())

df = pd.read_csv('cafe.csv')
print(df.head(2))

df['new'] = 0 
print(df.head(2))

df['할인가'] = df['가격'] * (1-0.2)

print(df.head())

df = pd.read_csv('cafe.csv')
print(df.head())

df.drop(1,axis=0 , inplace=True )
print(df.head())

df.drop('칼로리' , axis=1,inplace=True)

print(df.head())

print(df.shape)

<<<<<<< HEAD

df = pd.read_csv('date_data.csv')

print(df['DateTime2'])
df['Date2'] = pd.to_datetime(df['Date2'] , format="%Y-%m-%d")
df['Date3'] = pd.to_datetime(df['Date3'], format='%Y-%m-%d')
df['DateTime1'] = pd.to_datetime(df['DateTime1'] , format='%Y-%m-%d %H:%M:%S')
df['DateTime2'] = pd.to_datetime(df['DateTime2'] , format='%Y-%m-%d %H:%M:%S')

print(df[['Date2' , 'Date3' , 'DateTime1' , 'DateTime2']])
#df.drop(columns=['Unnamed: 0'] , inplace=True)
#df.to_csv('date_data.csv' , index=False)
#
# print(df)

print(df.info())

df['year'] = df['DateTime1'].dt.year
df['month'] = df['DateTime2'].dt.month
df['day'] = df['DateTime1'].dt.day
df['hour'] = df['DateTime1'].dt.hour
df['minute'] = df['DateTime1'].dt.minute
df['second'] = df['DateTime1'].dt.second

print(df)

df['날짜'] = df['DateTime1'].dt.dayofweek

print(df)

print(df['DateTime1'].dt.to_period("Y"))
print(df['DateTime1'].dt.to_period("Q"))
print(df['DateTime1'].dt.to_period("M"))

df = pd.read_csv("date_data.csv" , usecols=['DateTime4'] , parse_dates=['DateTime4'])

print(df)

day = pd.Timedelta(days=99)
df['100day'] = df['DateTime4'] + day
print(df)

day = pd.Timedelta(hours=100)
df['100hours'] = df['DateTime4'] + day

print(df)

diff = pd.Timedelta(weeks=7 , days=7 , hours=7 , minutes=7 , seconds=7)
df['diff'] = df['DateTime4'] - diff
print(df)


diff = df['100hours'] - df['diff']
print(diff)

print(diff.dt.total_seconds())
print(diff.dt.total_seconds()/60)
print(diff.dt.total_seconds()/60/60/24)

print(diff.dt.days)
print(diff.dt.seconds)

print(round(diff.dt.total_seconds()/60))

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

full_menu = pd.concat([appetizer , main] , ignore_index = True)
print(full_menu)

full_menu = pd.concat([appetizer, main] , axis=1)
print(full_menu)

price = pd.DataFrame({
    'Menu' : ['Salad' , 'Soup' , 'Steak' , 'Pasta'] ,
    'Price' : [5000,3000,15000,12000]
})

print(price)

cal = pd.DataFrame({
    'Menu' : ['Soup' , 'Steak','Pasta','Salad'],
    'Calories' : [100,500,400,150]
})

menu_info = pd.merge(price, cal, on="Menu")

print(menu_info)


data = {
    '이름' : ['서아' , '다인' , '채아', '예담' , '종현', '태헌'] ,
    '부서' : ['개발' , '기획' ,'개발', '기획' , '개발' , '기획'] ,
    '급여' :  [3000,3200,3100,3300,2900,3100]
}

df = pd.DataFrame(data)
print(df)

pt = df.pivot_table(index='부서' , values = '급여' , aggfunc='mean')
print(pt)

data = {
    '부서' : ['개발' , '기획' , '기획' , '기획' , '개발' , '개발'],
    '직급' : ['대리','과장' , '대리' , '과장' , '대리' , '과장'] ,
    '급여' : [3000,4000,3200,4200,3500,4500]
}

df = pd.DataFrame(data)

print(df)

pt = df.pivot_table(index='부서' , columns='직급' , values='급여' , aggfunc='sum')
print(pt)


data = {
    '부서' : ['개발' , '기획' , '기획' , '기획' , '개발' , '개발'],
    '성별' : ['남' , '여' , '남' , '여' , '남' , '여'] ,
    '근속연수' : [3,5,4,6,7,8]
}

df = pd.DataFrame(data)
print(df)

pt = df.pivot_table(index='부서' , columns='성별' , values='근속연수' , aggfunc='mean')

print(pt)

df = pd.DataFrame({
    "구분" : ['전자','전자','전자','전자','전자','가전','가전','가전','가전'],
    "유형" : ['일반','일반','일반','특수','특수','일반','일반','특수','특수'],
    "크기" : ['소형' , '대형' , '대형' , '소형' , '소형' , '대형' , '소형' , '소형' , '대형'],
    "수량" : [1,2,2,3,3,4,5,6,7],
    "금액" : [2,4,5,5,6,6,8,9,9]
})
print(df)

pt = df.pivot_table(index=['구분','유형'] , values='수량' , columns=['크기'] , aggfunc='sum')

print(pt)

pt = df.pivot_table(index=['구분','유형'] , values='수량' , columns=['크기'] , aggfunc='sum' , fill_value=0)
print(pt)


pt = pd.pivot_table(df, values=['수량' ,'금액'] , index=['구분','크기'] , aggfunc={'수량' : 'mean' , '금액':'mean'})

print(pt)

pt = pd.pivot_table(df ,values=['수량','금액'] , index=['구분','크기'] , aggfunc={'수량':'mean' , '금액': ['min','max','mean']})

print(pt)

print(pt.reset_index())


print('---------------------------------------------------------------------------------------------------------------')

import pandas as pd
import numpy as np

df = pd.DataFrame({
    '메뉴' : ['아메리카노' , '카페라떼' , '에스프레소', '카페모카' , '바닐라라떼'],
    '가격' : [4500,5000,4000,5900,5300],
    '칼로리' : [10,110 , np.NaN , 210 , np.NaN] ,
    '원두' : ['과테말라' , '브라질' , ' 과테말라' , np.NaN , np.NaN]
})

print(df)
#1
miv = df['칼로리'].min()

df['칼로리'] = df['칼로리'].fillna(miv)

print(df)

#2
mov = df['원두'].mode()[0]

df['원두'] = df['원두'].fillna(mov)

print(df)

#3 

cond = df['가격'] >= 5000

print(len(df[cond]))

#4

df['이벤트가'] = df['가격']*0.5

print(df)

#5

df.drop(columns=['칼로리'] , inplace=True)

print(df)

#6

print(df.loc[:2])

#7

print(df.iloc[:3])

#8

#df.drop(columns=['이벤트가','원두'], inplace=True)
print(df.loc[1:2 , :'가격'])
print(type(df.loc[1:2]))

#9

print(df.iloc[1:3 , :2])

#10
print('문제10')
print(df.sort_values('가격' , ascending=False).reset_index().loc[:2])
=======
df = pd.read_csv('cafe.csv')
print(df.loc[0])
print(df.loc[1,'가격'])

print(df.loc[:,'가격'])

print(df.loc[2,'메뉴':'칼로리'])

print(df.loc[2,['메뉴','칼로리']])

print(df.loc[1:3,'가격':'칼로리'])

df = pd.read_csv('cafe.csv')
df.drop(0,axis=0,inplace=True)
print(df.head())

print(df.iloc[0,1])
print(df.loc[1,'메뉴'])

print(df.iloc[2,:2])

print(df.iloc[1:3])

print(df.loc[:,'메뉴':'칼로리'])

import numpy as np

df = pd.read_csv('cafe.csv')
df['원산지']  = np.nan
print(df)

df.loc[0,'원산지']  = '콜롬비아'
df.loc[2:3,'원산지'] = '과테말라'
print(df.head())

df.loc['시즌'] = ['크리스마스라떼' , 6000,300,'한국']

print(df.head(8))

df.loc[7] = {'메뉴':'딴짓커피' , '가격':2000 , '칼로리':20}

print(df.tail())

df.drop("시즌",axis=0,inplace=True)

df.to_csv('cafe2.csv' , index=False)

df = pd.read_csv('cafe2.csv')
print(df.head())

print(df.sort_index(ascending=False))
print(df.sort_values('메뉴',ascending=False))

print(df.sort_values(['가격','메뉴'],ascending=[False,True],inplace=True))
print(df)

print(df.reset_index(drop=True))
>>>>>>> a78539af24f02cc4058a6ae4462cc5ba64959a82
