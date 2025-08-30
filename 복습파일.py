<<<<<<< HEAD
print('안녕 시나공')

print(2024)

print(16//3)

print(4**2)

print(int(4**0.5))

print(4+3*2)

print('py'+'thon')

print('py'+'3')

print(type(1.1))

print(type(False))

print(True+True+False+True)

a,b,c = 10,20,30

print(a+b+c)

score = 10

if score>9:
    print('9보다 크다')
    

print('True and False' , True and False)
print('True and True' , True and True)
print('True or False' , True or False)
print('True or True' , True or True)

if score < 10 and score >=5 :
    print('score는 5이상 10 미만입니다람쥐')
else:
    print('score는 5미만 10이상입니다람쥐')
    
score = 7

if score >=10:
    print('A')
elif score <10 and score >=5:
    print('B')
else:
    print('C')
    
    
score =5 
if score >= 10:
    print('score는 10이상입니다.')
elif score <10 and score >=5:
    print('score는 5이상 10미만입니다.')
elif score <4 and score >=3:
    print('score는 3 이상 4미만입니다람쥐')
else :
    print('score는 2이하입니다람쥐')
    
    
listbox = [4,2,10,6,8]
sorted_box = sorted(listbox)

print(sorted_box)

dictbox = {'name':'쿼카' , 'level':5}

print(dictbox)

print(dictbox['name'])

print(dictbox['level'])

dictbox['level'] = 9
print(dictbox)
print(dictbox.keys())
print(dictbox.values())

print(dictbox.items())

print(list(dictbox.values()))

print(list(dictbox.items()))
a = '합격맛집'
b = '길벗'
c = '출판사'
print({'닉네임' : a , c: b })

listbox1 = ['딴짓분식' , '딴짓카페' , '딴짓피자']
listbox2 = [4.8,4.9,5.0]

dictbox = {'가게' : listbox1 , '평점' : listbox2}

print(dictbox)

listbox = [2,4,6,8,10]

print(listbox[-1])

print(listbox[0:3])

print(listbox[::2])

listbox[1:3] = [10,20]

print(listbox)

listbox  = [4,2,10,6,8]

print(sum(listbox))

print(max(listbox))

print(len(listbox))

print(min(listbox))

print(round(1.2345,2))

text = '빅데이터분석기사 파이썬 공부'
text = text.replace('공부','스터디')
print(text)

text = '빅데이터분석기사 파이썬 공부'
text = text.replace('파이썬','머신러닝').replace('분석기사' , '분석을 위한')
print(text)

text = '안녕하세요! 함께 성장해요'
text[:2]

print(text[:2])
print(text[7:9])

date = '2022-12-25'
print(date[5:])

print(date.split('-'))

print(list(date))

for item in listbox :
    print(item)
    
listbox = [2,4,6,8,10]

for item in listbox:
    result = item +1
    print(result)
    
for item in range(1,6):
    print(item)
    
    
def hello():
    print('하이요')

hello()
hello()
hello()

def hello(e):
    print('hello' , e)
    
hello('빅분기')


def plus(x,y):
    print(x+y)
    
plus(2,3)


def plus2(x,y):
    result = x+y
    return result

a = plus2(2,3)
print(a)


listbox = ['감사','행복','사랑','성공','긍정','변화','성장','희망']

print(len(listbox))
print(listbox[0])
print(listbox[-1])
print(listbox[:3])
print(listbox[-2])

print(listbox[1:3])
listbox[5] = '웃음'
print(listbox)

cols = ['name' , 'age','phone']
for i in cols:
    print(i)
    
listbox = [15,46,78,24,56]

def maxmin(e):
    result = max(e)-min(e)
    return result
a = maxmin(listbox)
print(a)

def replaceit(e):
    result = e.replace('여러분','당신')
    return result

a = replaceit('여러분의 합격을 응원합니다.')
print(a)
=======
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
>>>>>>> a78539af24f02cc4058a6ae4462cc5ba64959a82
