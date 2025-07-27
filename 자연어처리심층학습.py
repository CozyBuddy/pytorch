from random import Random
import torch

print(torch.tensor([1,2,3]))
print(torch.Tensor([[1,2,3],[4,5,6]]))
print(torch.LongTensor([1,2,3]))
print(torch.FloatTensor([1,2,3]))

tensor = torch.rand(1,2) # 1행 2열 

print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

print(tensor)
print(tensor.shape)

tensor = tensor.reshape(2,1)

print(tensor)
print(tensor.shape)

tensor = torch.rand((3,3) ,dtype=torch.float)
print(tensor)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cpu = torch.FloatTensor([1,2,3])
gpu = torch.cuda.FloatTensor([1,2,3])

tensor = torch.rand((1,1), device=device)

print(device)
print(cpu)
print(gpu)
print(tensor)

cpu = torch.FloatTensor([1,2,3])
gpu = cpu.cuda()
gpu2cpu = gpu.cpu()
cpu2gpu = cpu.to('cuda')

print(cpu)
print(gpu)
print(gpu2cpu)
print(cpu2gpu)

import numpy as np

ndarray = np.array([1,2,3] ,dtype=np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))

tensor = torch.cuda.FloatTensor([1,2,3])
ndarray = tensor.detach().cpu().numpy()
print(ndarray)
print(type(ndarray))

import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

man_height = stats.norm.rvs(loc=170 , scale=10 , size=500 , random_state=1) # 특정 평균 (loc) . 표준편차 (scale) , 샘플링데이터(size)
woman_height = stats.norm.rvs(loc=150 , scale=10 , size=500 , random_state=1)

X = np.concatenate([man_height , woman_height])
Y = ['man'] * len(man_height) + ['woman'] * len(woman_height)

df = pd.DataFrame(list(zip(X,Y)) , columns=['X','Y'])

print(df)
fig = sns.displot(data=df, x='X' , hue='Y' , kind='kde')
fig.set_axis_labels('cm','count')

plt.show()

statstic , pvalue = stats.ttest_ind(man_height , woman_height , equal_var=True)

print('statstic' , statstic)
print('pvalue' , pvalue)
print('*:' , pvalue < 0.05)
print('**:' , pvalue < 0.001)


x = []

for i in range(1,30):
    x.append([i])

print(x)

x = np.array(x)

y = []

import random
for i in range(1,30):
    y.append(round(random.uniform(1,30),2))

print(y)

y = np.array(y)

weight = 0.0
bias = 0.0
learning_rate = 0.001

# for epoch in range(10000):



for epoch in range(10000):
    y_hat = weight * x + bias
    cost = ((y -y_hat)**2).mean()


    weight = weight - learning_rate * ((y_hat - y) * x).mean()
    bias = bias - learning_rate * (y_hat - y).mean()

    if(epoch+1) % 1000 == 0:
        print(f"Epoch : {epoch+1 :4d} , Weight :{weight : .3f} , Bias : {bias:.3f} ,Cost:{cost:.3f}")

