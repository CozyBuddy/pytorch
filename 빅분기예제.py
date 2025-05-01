import numpy as np
import pandas as pd

a = np.array([1, 2, 3])    # 1차원 배열
b = np.array([[1, 2], [3, 4]])  # 2차원 배열


# a + 1        # [2 3 4]
# a * 2        # [2 4 6]
# a ** 2       # [1 4 9]

# print(a + 1  )
# print(a * 2)
# print(a ** 2 )

# print(b.reshape(4))
# print(b.reshape(1,4))

# print(b[0, 1]  ,b[:, 1]    ,b[1, :]    )



# data = {'name': ['Alice', 'Bob', 'Charlie'],
#         'age': [25, 30, 35]}

# df = pd.DataFrame(data)

train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

# print(train.head(100))

df2 = train[train['Survived'] == 1]
#print(df2.head(100))
# print(train['Survived'].mean())

# 점수(score) 기준 내림차순 정렬
sorted_df = train.sort_values(by='Pclass', ascending=True)

sorted_df = train.sort_values(by='Pclass' , ascending=False)
sorted_df['Count'] = train['Survived'] == 1
# print(sorted_df)
# print(sorted_df['Cabin'].unique())

print(sorted_df.dropna(subset=['Cabin']).sort_values(by='PassengerId'))

# print(sorted_df.dropna())
#print(sorted_df.fillna(sorted_df['Cabin']))

#dir