import torch
import torch.nn as nn
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

dataset = pd.read_csv('car_evaluation.csv')

print(dataset.head())
print(dataset.shape)

fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 8
fig_size[1] = 6

plt.rcParams['figure.figsize'] = fig_size

#dataset['output'].value_counts().plot(kind='bar', color=['lightblue' , 'lightgreen' , 'orange' , 'pink'] )
dataset['output'].value_counts().plot(kind='pie' , autopct='%0.02f', colors=['lightblue' , 'lightgreen' , 'orange' , 'pink'] , explode=[0.05,0.05,0.05,0.05])
plt.show()

categorical_columns = ['price' , 'maint' , 'doors' ,'persons' ,'lug_capacity' , 'safety']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')

price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values

persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price,maint, doors, persons ,lug_capacity , safety] ,1) # 열끼리 붙여서 차원을 높임

print(categorical_data[:10])

categorical_data = torch.tensor(categorical_data ,dtype=torch.int64)


outputs = pd.get_dummies(dataset['output'])
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

print(outputs.shape)