import os 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset ,DataLoader
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split

data1 = pd.read_csv('seoul_air_checkData.csv' , encoding='euc-kr')
data2 = pd.read_csv('seoul_air_checkLocation.csv', encoding='euc-kr')
data3 = pd.read_csv('seoul_air_hour_2022.csv')

print(data1.head())
print(data2.head())
print(data3.head())

df = pd.merge(data3 ,data2 , on='측정소 코드')

df = pd.merge(df,data1 , left_on='측정항목' , right_on='측정항목 코드')
print(df.head())

df = df[['측정일시' , '측정소 이름' ,'측정기 상태' , '측정항목' , '평균값']]

cond = df['측정항목'] == 9
cond3 = (df['평균값'] >=0) & (df['평균값'] <=300)
cond4 = df['측정소 이름'] == '강남구'
cond5 = df['측정일시'] %100 == 0
df = df[cond]
df = df[cond3]
df = df[cond4]
df = df[cond5]

print(df.head())
df['측정일시'] = df['측정일시']/100
df['측정일시'] = pd.to_datetime(df['측정일시'] ,format='%Y%m%d')
print(df.head())

df['측정소이름'] = df['측정소 이름']

df.drop(columns=['측정소 이름'] , inplace=True)

cond2 = df['측정기 상태'] == 0
df = df[cond]
df.drop(columns=['측정기 상태','측정항목'] , inplace=True)

print(df.head())


# print(df.head())

#df['측정일시'] = pd.to_datetime(df['측정일시'])

df.set_index('측정일시',inplace=True)

print(df.info())
# df['Volume'] = df['Volume'].astype(float)

trainC = int(len(df) * 0.9)
X = df.iloc[: , 1]
y = df.iloc[: , [0]]
print(X.head())
print(y.head())

# # 스케일링
ms = MinMaxScaler()

X_scaled = pd.get_dummies(X)
y_scaled = ms.fit_transform(y)  

# # 훈련/테스트 분할
X_train, X_test = X_scaled[:trainC], X_scaled[trainC:]
y_train, y_test = y_scaled[:trainC], y_scaled[trainC:]

print(X_train.shape, y_train.shape)

# # 텐서로 변환 + 3D 입력 형식 맞추기 (LSTM용)
X_train_tensors_f = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  # (batch, seq, features) batch_size가 데이터갯수 200이고 seq 가 1 이되고 features가 4가됨.
X_test_tensors_f = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)

y_train_tensors = torch.tensor(y_train, dtype=torch.float32)
y_test_tensors = torch.tensor(y_test, dtype=torch.float32)

# #### unsqueeze
# #  x= [1,2,3] 일때 unsqueeze(-1) 하면 맨 뒤에 차원을 추가 , 즉 지금 shape이 3 이므로 3,1 3행 1열 이됨
# #  unsqueeze(0) 이면 맨 앞에 차원을  추가해서 1행3열이 되고 , unsqueeze(1)은 shape 이 3이므로 바로 3다음에 차원을 추가해서 3행 1열이됨.


# ## seq_length 이거는 LSTM 에서 과거의 데이터를 몇개 기억하고 있을건지 정하는 숫자
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size , hidden_size , num_layers , seq_length):
        super(LSTM , self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size , hidden_size=hidden_size , num_layers=num_layers , batch_first=True)

        self.fc_1 = nn.Linear(hidden_size ,128)
        self.fc = nn.Linear(128 , num_classes)
        self.relu = nn.ReLU()


    def forward(self , x):
        #print(x.size(0))
        h_0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_size) ## 0을 가득한 tensor을 생성할때 순서는 고정 x.size(0) 은 batch_size 임
        c_0  =torch.zeros(self.num_layers , x.size(0) , self.hidden_size)
        output , (hn ,cn) = self.lstm(x , (h_0 ,  c_0)) ## 문법상 이렇게 변수를 줘야함.
        hn = hn[-1] 
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    

num_epochs = 1000
learning_rate = 0.0001

input_size = 1 ## 독립변수 갯수
hidden_size = 15 ## 각 LSTM 층 안에서 한 시점마다 기억하는 벡터(은닉 상태)의 크기(길이) LSTM이 스스로 어떤 걸 기억할지 결정해서 업데이트 하는공간
num_layers = 3 ## 은닉층 갯수

num_classes = 1 ## 회귀문제에서는 항상 0 분류문제에서는 분류되는 클래스 갯수를 넣음

model = LSTM(num_classes , input_size , hidden_size , num_layers , X_train_tensors_f.shape[1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model.forward(X_train_tensors_f)
    optimizer.zero_grad()
    loss = criterion(outputs , y_train_tensors)
    loss.backward()

    optimizer.step()
    if epoch % 100 == 0 :
        print('Epoch : %d , loss : %1.5f' % (epoch , loss.item()))


# #df_x_ss = ss.transform(df.iloc[: , :-1])
df_y_ms = ms.transform(df.iloc[: , [0]])

# #df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = torch.tensor(df_y_ms)

# df_x_ss = torch.reshape(df_x_ss , (df_x_ss.shape[0] ,1 , df_x_ss.shape[1]))
df_x = torch.tensor(X_scaled.values , dtype=torch.float32).unsqueeze(1)

train_predict = model(df_x)
predicted = train_predict.data.numpy() 

predicted_inv = ms.inverse_transform(predicted)  # y 스케일 역변환
label_y_inv = ms.inverse_transform(df_y_ms.numpy())  # 실제 y값도 역변환


plt.figure(figsize=(12,6))
plt.plot(label_y_inv, label='Actual Data')
plt.plot(predicted_inv, label='Predicted Data')
plt.xlabel('Time (Days)')
plt.ylabel('미세먼지 농도')
plt.title('일자별 미세먼지 예측 vs 실제값')
plt.legend()
plt.show()

# train_predict = model(df_x_ss)
# predicted = train_predict.data.numpy()
# label_y = df_y_ms.data.numpy()

# predicted = ms.inverse_transform(predicted)
# label_y = ms.inverse_transform(label_y)

# plt.figure(figsize=(10,6))
# plt.axvline(x=200 , c='r' , linestyle='--')

# plt.plot(label_y , label='Actual_data')
# plt.plot(predicted , label='Predicted Data')

# plt.title("Time Series Prediction")
# plt.legend()
# plt.show()












