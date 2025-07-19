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

import yfinance as yf

# 예: 애플(AAPL) 주가 데이터 가져오기
# data = yf.download('SBUX', start='2024-01-01', end='2024-12-31')
# # 과거 5년 데이터
# #data = ticker.history(period='1y')

# # CSV로 저장
# data.to_csv('SBUX.csv')

# print('CSV 저장 완료!')


df = pd.read_csv('SBUX.csv')

print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date',inplace=True)

df['Volume'] = df['Volume'].astype(float)

X = df.iloc[: , :-1]
y = df.iloc[: , 4:5]
print(X.head())
print(y.head())

# 스케일링
ss = StandardScaler()
ms = MinMaxScaler()

X_scaled = ss.fit_transform(df.iloc[:, :-1])
y_scaled = ms.fit_transform(df.iloc[:, 4:5])  # 'Close' 값

# 훈련/테스트 분할
X_train, X_test = X_scaled[:200], X_scaled[200:]
y_train, y_test = y_scaled[:200], y_scaled[200:]

print(X_train.shape, y_train.shape)

# 텐서로 변환 + 3D 입력 형식 맞추기 (LSTM용)
X_train_tensors_f = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch, seq, features)
X_test_tensors_f = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train_tensors = torch.tensor(y_train, dtype=torch.float32)
y_test_tensors = torch.tensor(y_test, dtype=torch.float32)

#### unsqueeze
#  x= [1,2,3] 일때 unsqueeze(-1) 하면 맨 뒤에 차원을 추가 , 즉 지금 shape이 3 이므로 3,1 3행 1열 이됨
#  unsqueeze(0) 이면 맨 앞에 차원을  추가해서 1행3열이 되고 , unsqueeze(1)은 shape 이 3이므로 바로 3다음에 차원을 추가해서 3행 1열이됨.

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
        h_0 = Variable(torch.zeros(self.num_layers , x.size(0) , self.hidden_size))
        c_0  =Variable(torch.zeros(self.num_layers , x.size(0) , self.hidden_size))
        output , (hn ,cn) = self.lstm(x , (h_0 ,  c_0))
        hn = hn[-1] 
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    

num_epochs = 10000
learning_rate = 0.0001

input_size = 4
hidden_size = 16
num_layers = 1

num_classes = 1

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


df_x_ss = ss.transform(df.iloc[: , :-1])
df_y_ms = ms.transform(df.iloc[: , -1:])

df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = Variable(torch.Tensor(df_y_ms))

df_x_ss = torch.reshape(df_x_ss , (df_x_ss.shape[0] ,1 , df_x_ss.shape[1]))

train_predict = model(df_x_ss)
predicted = train_predict.data.numpy()
label_y = df_y_ms.data.numpy()

predicted = ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)

plt.figure(figsize=(10,6))
plt.axvline(x=200 , c='r' , linestyle='--')

plt.plot(label_y , label='Actual_data')
plt.plot(predicted , label='Predicted Data')

plt.title("Time Series Prediction")
plt.legend()
plt.show()












