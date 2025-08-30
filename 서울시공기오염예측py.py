import pandas as pd

data1 = pd.read_csv('seoul_air_checkData.csv' , encoding='euc-kr')
data2 = pd.read_csv('seoul_air_checkLocation.csv', encoding='euc-kr')
data3 = pd.read_csv('seoul_air_hour_2022.csv')

# print(data1.head())
# print(data2.head())
# print(data3.head())


# print(data1.info())
# print(data2.info())
# print(data3.info())


print(data3.describe())
merge_data = pd.merge(data2,data3 , on='측정소 코드')

print(merge_data.head())

mdata = pd.merge(merge_data , data1 , left_on='측정항목' , right_on='측정항목 코드')

print(mdata.head())

print(mdata.info())

print(mdata['측정항목 코드'])
cond = (mdata['측정항목 코드'] == 9) & (mdata['평균값'] >=0)

mdata = mdata[cond]

print(mdata.head())

#mdata = mdata[['측정일시' ,'측정소 코드','측정항목','평균값','측정기 상태','국가 기준초과 구분','지자체 기준초과 구분','측정소 이름']]
mdata = mdata[['측정일시' ,'평균값','측정소 이름']]

target = mdata.pop('평균값')



mdata['측정일시'] = pd.to_datetime(mdata['측정일시'], format='%Y%m%d%H')

mdata['년도'] = mdata['측정일시'].dt.year 
mdata['월'] = mdata['측정일시'].dt.month 
mdata['일'] = mdata['측정일시'].dt.day 
mdata['시간'] = mdata['측정일시'].dt.hour

mdata = mdata.drop(columns='측정일시')


data = pd.get_dummies(mdata)
from sklearn.model_selection import train_test_split

X_train , X_val , y_train ,y_val = train_test_split(data,target , test_size=0.1 , random_state=0)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_val)

from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_val , y_pred)

print('rmse' , rmse)

print( 'y_val' , y_val.values)
print( 'y_pred' , y_pred)
print('max y_pred' , y_pred.max())
import joblib
joblib.dump(rf, 'seoul_poll_model.pkl', compress=6)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6,6))
plt.plot(np.arange(len(y_val)), y_val.values ,label='Actual', color='blue', alpha=0.6)
plt.plot(np.arange(len(y_pred)), y_pred,label='Predicted', color='red', alpha=0.6)

#plt.scatter(y_val, y_pred, alpha=0.6, color='blue')
# plt.xscale('log')
# plt.yscale('log')
#plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')  # y = x 선
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
