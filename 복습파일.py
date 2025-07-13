import pandas as pd

df = pd.DataFrame({
    'Name' : { 0: '쿼카' , 1 : '알파카' , 2: '시바견'},
    '수학' : {0:90 , 1 :93 , 2:85} ,
    '영어' : {0 : 92 , 1:84 , 2:86} ,
    '국어' : {0:91 ,1 :94 ,2:83}
})

melted_df = pd.melt(df , id_vars=['Name'] , var_name='과목' , value_name='점수')

print(melted_df.head())

df = pd.DataFrame({
    '반' : {0:'A반' , 1:'A반' , 2:'B반'},
    '이름' : { 0: '쿼카' , 1 : '알파카' , 2: '시바견'},
    '수학' : {0:90 , 1 :93 , 2:85} ,
    '영어' : {0 : 92 , 1:84 , 2:86} ,
    '국어' : {0:91 ,1 :94 ,2:83}
})

df_edge = pd.melt(df ,id_vars=['반','이름'] , var_name='과목' , value_name='점수')
print(df_edge)

df = pd.read_csv('cafe4.csv')

print(df.head())


print(df.groupby(['원산지']).mean(numeric_only=True))

print(df.groupby(['원산지','칼로리']).mean(numeric_only=True).reset_index())

df = pd.DataFrame({
    '과일' : ['딸기' , '블루베리' , '딸기' , '블루베리' , '딸기' , '블루베리' , '딸기' , '블루베리'],
    '가격' : [1000,None ,1500,None ,2000,2500,None, 1800]
})

price = df.groupby('과일')['가격'].transform('mean')

print(price.head(10))

df['가격'] = df['가격'].fillna(price)

print(df)

df = pd.DataFrame({
    '과일' : ['딸기' , '블루베리' , '딸기' , '블루베리' , '딸기' , '블루베리' , '딸기' , '블루베리'],
    '등급' : ['B' ,'B' , 'A' , 'A','A','A' ,'B','B'],
    '가격' : [1000,None ,1500,None,2000,2500,None,1800]
})


price = df.groupby(['과일','등급'])['가격'].transform('mean')
print(price)

df['가격'] = df['가격'].fillna(price)

print(df)