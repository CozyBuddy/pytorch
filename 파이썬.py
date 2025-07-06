# # # # # print('안녕 시나공')
# # # # # print(2024)
# # # # # #길벗은
# # # # # # print("파이썬은쉽다")
# # # # # print("파이썬은어렵다")

# # # # # print(1+2,4-1,2*3,16/3)
# # # # # print(15//3)
# # # # # print(16//3)
# # # # # print(16%3)
# # # # print(14**2)
# # # # print(16**0.5)
# # # # print('py'+'thon')
# # # # print('py'+"3")
# # # # print(type(1))
# # # # print(type('1.1'))

# # # # print(type(3),type(3.0),type('3'))

# # # # print(type(True), type(False))

# # # # print(True+True+False+True+False)

# # # # 변수명 설정 > 대소문자 구분 예약어 사용불가 

# # # americano = 4500 
# # # print(americano)

# # # # americano = 'americano'

# # # # print(americano)
# # # lattee = 5000
# # # print(americano+lattee)

# # # result  = americano + lattee

# # # print(result)

# # americano = 4500
# # americano = americano - 1000
# # print(americano)

# # a,b,c  =  10,20,30
# # print(a,b,c)

# # a=2
# # b=0.17
# # c=a+b
# # print(type(c))

# # box = 2.17
# # print("문자열" ,str(box))
# # print("정수형",int(box))

# # box = "3.14"
# # box = float(box)+10
# # print(box)
# # print(10<5)
# # print(10>5)

# # print(10>=5)

# # print(10==10)
# # print(10 !=5)

# # a=5
# # b=10
# # print(a==b)
# # print(a>=b)
# # print(a<=b)

# # c = "빅데이터"
# # d = "빅데이터"
# # e = "데이터"

# # print(c==d)
# # print(c==e)

# 점수=77
# if 점수 >=90:
#     print("A")
# elif 점수 >= 80:
#     print("B")
# elif 점수 >= 70:
#     print("C")
# else:
#     print("D")

# if True:
#     print("실행")

# if False:
#     print("무시")

# score = 10
# if score >=10:
#     print("10보다 크거나 같다")

# if score>9:
#     print("9보다 크다")

# if score<=10:
#     print("10보다 작거나 같다.")

# if score <11:
#     print("11보다 작다")

# if score==10:
#     print("score는 10이다")

# if score!=11:
#     print("score는 11과 같지 않다.")

# print("True and False:" , True and False)
# print("True and True:" , True and True)
# print("True or False:" , True or False)
# print("True or True:" , True or False)

# if score<10 and score >=5:
#     print("score는 5이상 , 10미만입니다.")
# else:
#     print("score는 5미만, 10 이상입니다.")

# score = 7 
# if score >=10:
#     print("A")
# elif score<10 and score >=5:
#     print("B")
# else:
#     print("C")


# listbox = [4,2,10,6,8]
# print(listbox)

# print(listbox[1])
# print(type(listbox))

# listbox = ['길벗','시나공','빅데이터']
# print(listbox)

# print(listbox.append("분석"))
# print(listbox)

# listbox = [100,200,500,1,2,3,4,5,10]
# print(sorted(listbox , reverse=True))
# print(sorted(listbox))

# dictbox = {'name' : '쿼카' , 'level': 5}

# print(dictbox)
# print(type(dictbox))
# print(dictbox['name'])
# print(dictbox['level'])

# dictbox['level'] = 6
# print(dictbox['level'])

# print(dictbox.keys())
# print(dictbox.values())
# print(dictbox.items())

# print(list(dictbox.values()))
# a='합격맛집'
# b='길벗'
# c='출판사'

# dictbox = { '닉네임': a , c :b }
# print(dictbox)

# listbox = ['딴짓분식','딴짓카페','딴짓피자']
# listbox2 = [4.8,4.9,5.0]

# dictbox = {'가게': listbox , '평점': listbox2}
# print(dictbox)

# listbox = [2,4,6,8,10]
# print(listbox[0])
# print(listbox[3])
# print(listbox[-1])
# print(listbox[-2])

# print(listbox[0:3])
# print(listbox[1:3])
# print(listbox[3:])
# print(listbox[:3])

# print(listbox[::2])

# listbox[1:3] = [10,20]
# print(listbox)

# listbox = [4,2,10,6,8]
# print(sum(listbox))

# boolbox = [True,False,True]
# print(sum(boolbox))
# print(max(listbox))

# print(min(listbox))

# print(len(listbox))

# print(round(1.2345,2) , round(1.2375,2))

# text = '빅데이터 분석기사 파이썬 공부'
# text = text.replace("공부" , "스터디")

# text = text.replace("파이썬","머신러닝").replace("분석기사","분석을 위한")
# print(text)

text = '안녕하세요! 함께 성장해요.'
print(text[:2])

print(text[7:9])

date = "2022-12-25"
print(date[5:])


print(date.split('-'))
print(text.split(' '))

print(list(date))

listbox = [2,4,6,8,10]
for item in listbox:
    print(item)

listbox = ['길벗','시나공','빅분기']
for item in listbox:
    print(item)
    print(1)
print('당연한거아님?')

for i in range(1,10):
    print(i)

listbox  = []

for i in range(1,6):
    listbox.append(i)
    print(listbox)

listbox = [ '길벗','시나공','빅분기','분석']
for i,it in enumerate(listbox):
    print(i,it)

person_info = { 'name':'사랑' , 'age':20 , 'city':'부산' , 'hobbies': ['연애','수영','코딩']}

for k,v in zip(person_info.keys(),person_info.values()):
    print(k,v)

def hello():
    print("안녕하세요")


hello()
hello()
hello()

def hello(name):
    print('hello'+name)

hello('날따라')

def plus(x,y):
    print(x+y)


plus(10,20)

def plust(x,y):
    result = x+y
    return result

a = plust(2,3)
print(a)

listbox = [15,46,78,24,56]
def min_max(data):
    mi = min(data)
    ma = max(data)
    return mi,ma

a,b = min_max(listbox)
print(a,b)

def mean(data):
    return sum(data)/len(data)

print(mean(listbox))

print("------------------------------------------------------------------------------------")
# 확인문제 p.65
listbox = ["감사","행복","사랑","성공","긍정","변화","성장","희망"]

print(len(listbox))
print(listbox[0])
print(listbox[-1])
print(listbox[:3])
print(listbox[-2])
print(listbox[1:3])
listbox[5]='웃음'
print(listbox)

cols = ['name','age','phone']
for i in cols:
    print(i)

listbox=[15,46,78,24,56]

def mm(data):
    ma = max(data)
    mi = min(data)

    return ma - mi

print(mm(listbox))


str_data = '여러분의 합격을 응원합니다!'

def chstr(str):
    str = str.replace('여러분','당신')
    return str


print(chstr(str_data))


