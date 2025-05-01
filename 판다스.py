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