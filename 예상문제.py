import pandas as pd


import pandas as pd

data = {
    '제품명': ['A제품', 'B제품', 'C제품'],
    '2024Q1': [1200, 1500, 1100],
    '2024Q2': [1300, 1600, 1150],
    '2024Q3': [1250, 1580, 1130],
    '2024Q4': [1400, 1700, 1200]
}

df = pd.DataFrame(data)
print(df)

print(df.melt(id_vars=['제품명'] , var_name='분기' , value_name='매출'))

df = df.melt(id_vars=['제품명'] , var_name='분기' , value_name='매출')

print(df.pivot_table( index=['분기'], values='매출' , aggfunc='sum'))


import pandas as pd

# 원본 데이터
data = {
    '지역': ['서울', '서울', '부산', '부산'],
    '제품명': ['A제품', 'B제품', 'A제품', 'B제품'],
    '2024Q1': [500, 700, 400, 600],
    '2024Q2': [600, 750, 450, 620],
    '2024Q3': [550, 730, 480, 650],
    '2024Q4': [700, 800, 500, 670]
}

df = pd.DataFrame(data)

print(df.melt(id_vars=['지역','제품명'] , var_name='분기' ))
df = df.melt(id_vars=['지역','제품명'] , var_name='분기' )
print(df.pivot_table(index='지역' , columns='분기' ,values='value', aggfunc='sum'))


import pandas as pd

# 원본 데이터
import pandas as pd

data = {
    '학급': ['1반']*3 + ['2반']*3 + ['3반']*3 + ['4반']*3,
    '학생명': [
        '홍길동', '김철수', '박영희', 
        '이영희', '박민수', '최수진', 
        '강민석', '서지현', '윤하늘', 
        '조성민', '한지우', '임채영'
    ],
    '2025-01': [85, 90, 78, 88, 85, 92, 75, 80, 85, 90, 88, 87],
    '2025-02': [88, 92, 81, 85, 88, 90, 78, 82, 86, 89, 90, 88],
    '2025-03': [90, 91, 79, 87, 85, 89, 80, 85, 88, 92, 91, 90],
    '2025-04': [92, 89, 82, 90, 87, 91, 82, 87, 90, 91, 93, 92],
    '2025-05': [93, 90, 84, 91, 89, 92, 85, 89, 91, 94, 95, 93]
}

df = pd.DataFrame(data)
print(df)

df2 = df.groupby(['학급']).mean(numeric_only=True)
df2['평균차이'] = df2['2025-05'] - df2['2025-01']
print(df2)