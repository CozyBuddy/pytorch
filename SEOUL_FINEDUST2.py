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
import joblib
data1 = pd.read_csv('서울시 대기질 자료 제공_2008-2011.csv' , encoding='euc-kr')
data2 = pd.read_csv('서울시 대기질 자료 제공_2012-2015.csv', encoding='euc-kr')
data3 = pd.read_csv('서울시 대기질 자료 제공_2016-2019.csv' , encoding='euc-kr')
data4 = pd.read_csv('서울시 대기질 자료 제공_2020-2021.csv' , encoding='euc-kr')
data5 = pd.read_csv('서울시 대기질 자료 제공_2022.csv' , encoding='euc-kr')
data6 = pd.read_csv('서울시 대기질 자료 제공_2023.csv' , encoding='euc-kr')
data7 = pd.read_csv('서울시 대기질 자료 제공_2024.csv' , encoding='euc-kr')

print(data1.info())
print(data1.head())
print(data2.head())
print(data3.head())
print(data4.head())
print(data5.head())
print(data6.head())
print(data7.head())

data5['초미세먼지(PM25)'] = data5['초미세먼지(PM2.5)']

data5.drop(columns=['초미세먼지(PM2.5)'],inplace=True)
cond = data1['구분'] == '평균'
cond2 = data2['구분'] == '평균'
cond3 = data3['구분'] == '평균'
cond4 = data4['구분'] == '평균'
cond5 = data5['구분'] == '평균'
cond6 = data6['구분'] == '평균'
cond7 = data7['구분'] == '평균'

data1 = data1[cond]
data2 = data2[cond2]
data3 = data3[cond3]
data4 = data4[cond4]
data5 = data5[cond5]
data6 = data6[cond6]
data7 = data7[cond7]
df = pd.concat([data1,data2,data3,data4,data5,data6,data7])

print(df.head())
df['일시'] = pd.to_datetime(df['일시'])

print(df.info())
print(df.head())

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
df['year'] = df['일시'].dt.year
df['month'] = df['일시'].dt.month
df['day'] = df['일시'].dt.day
df['hour'] = df['일시'].dt.hour
df['dayofweek'] = df['일시'].dt.dayofweek

df['month_cos'] =  np.cos( 2 * np.pi*df['month']/12)
df['month_sin'] =  np.sin( 2 * np.pi*df['month']/12)

df['day_cos'] =  np.cos( 2 * np.pi*df['day']/31)
df['day_sin'] =  np.sin( 2 * np.pi*df['day']/31)


df['hour_cos'] =  np.cos( 2 * np.pi*df['hour']/24)
df['hour_sin'] =  np.sin( 2 * np.pi*df['hour']/24)

df['dayofweek_cos'] =  np.cos( 2 * np.pi*df['dayofweek']/7)
df['dayofweek_sin'] =  np.sin( 2 * np.pi*df['dayofweek']/7)

df.drop(columns=['year','day','hour','month','dayofweek','일시','구분','미세먼지(PM10)'],inplace=True)

print(df.head())
#df['측정일시'] = pd.to_datetime(df['측정일시'])

#df.set_index('일시',inplace=True)

#print(df.info())
# df['Volume'] = df['Volume'].astype(float)

trainC = int(len(df) * 0.9)
X = df.iloc[: , 1:]
y = df.iloc[: , [0]]
print(X.head())
print(y.head())

# # 스케일링
ms = MinMaxScaler(feature_range=(0, 10))
ss = StandardScaler()

X_scaled = ss.fit_transform(X)
y_scaled = ms.fit_transform(y)  

joblib.dump(ss, 'fine_scaler.pkl')
joblib.dump(ms, 'fine_scaler2.pkl')
def create_seq(x,y, seq_len):
    xs = []
    ys =[]
    for i in range(len(x) - seq_len):
        x_seq = x[i:(i+seq_len)]
        y_seq = y[i+seq_len]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs) , np.array(ys)
    
X_scaled , y_scaled= create_seq(X_scaled , y_scaled ,1)
# # 훈련/테스트 분할
X_train, X_test = X_scaled[:trainC], X_scaled[trainC:]
y_train, y_test = y_scaled[:trainC], y_scaled[trainC:]

print(X_train.shape, y_train.shape)

# # 텐서로 변환 + 3D 입력 형식 맞추기 (LSTM용)
X_train_tensors_f = torch.tensor(X_train, dtype=torch.float32).to(device)  # (batch, seq, features) batch_size가 데이터갯수 200이고 seq 가 1 이되고 features가 4가됨.
X_test_tensors_f = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train_tensors = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensors = torch.tensor(y_test, dtype=torch.float32).to(device)

# #### unsqueeze
# #  x= [1,2,3] 일때 unsqueeze(-1) 하면 맨 뒤에 차원을 추가 , 즉 지금 shape이 3 이므로 3,1 3행 1열 이됨
# #  unsqueeze(0) 이면 맨 앞에 차원을  추가해서 1행3열이 되고 , unsqueeze(1)은 shape 이 3이므로 바로 3다음에 차원을 추가해서 3행 1열이됨.


# ## seq_length 이거는 LSTM 에서 과거의 데이터를 몇개 기억하고 있을건지 정하는 숫자
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size , hidden_size , num_layers , seq_length):
        super(LSTM , self).__init__()
        # self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size , hidden_size=hidden_size , num_layers=num_layers , batch_first=True)

        self.fc_1 = nn.Linear(hidden_size ,512)
        self.fc = nn.Linear(512 , num_classes)
        self.relu = nn.ReLU()


    def forward(self , x):
        #print(x.size(0))
        h_0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_size).to(device) ## 0을 가득한 tensor을 생성할때 순서는 고정 x.size(0) 은 batch_size 임
        c_0  =torch.zeros(self.num_layers , x.size(0) , self.hidden_size).to(device)
        output , (hn ,cn) = self.lstm(x , (h_0 ,  c_0)) ## 문법상 이렇게 변수를 줘야함.
        hn = hn[-1] 
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    

num_epochs = 10000
learning_rate = 0.005

input_size = 8 ## 독립변수 갯수
hidden_size = 10 ## 각 LSTM 층 안에서 한 시점마다 기억하는 벡터(은닉 상태)의 크기(길이) LSTM이 스스로 어떤 걸 기억할지 결정해서 업데이트 하는공간
num_layers = 1 ## 은닉층 갯수

num_classes = 1 ## 회귀문제에서는 항상 0 분류문제에서는 분류되는 클래스 갯수를 넣음

model = LSTM(num_classes , input_size , hidden_size , num_layers , X_train_tensors_f.shape[1]).to(device)

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model.forward(X_train_tensors_f)
    optimizer.zero_grad()
    loss = criterion(outputs , y_train_tensors)
    loss.backward()

    optimizer.step()
    if epoch % 100 == 0 :
        print('Epoch : %d , loss : %1.5f' % (epoch , loss.item()))


df_x_ss = ss.transform(df.iloc[: , 1:])
df_y_ms = ms.transform(df.iloc[: , [0]])

df_x_ss , df_y_ms= create_seq(df_x_ss , df_y_ms , 1)

df_x_ss = torch.tensor(df_x_ss ,dtype=torch.float32).to(device)
df_y_ms = torch.tensor(df_y_ms,dtype=torch.float32).to(device)

# df_x_ss = torch.reshape(df_x_ss , (df_x_ss.shape[0] ,1 , df_x_ss.shape[1]))
# df_x = torch.tensor(X_scaled.values , dtype=torch.float32).unsqueeze(1)

train_predict = model(df_x_ss).to(device)
predicted = train_predict.cpu().data.numpy() 

predicted_inv = ms.inverse_transform(predicted)  # y 스케일 역변환
print(df_y_ms.cpu().numpy())
label_y_inv = ms.inverse_transform(df_y_ms.cpu().numpy().reshape(-1, 1))  # 실제 y값도 역변환

torch.save(model.state_dict(), 'finedust.pth')

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












