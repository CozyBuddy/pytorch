import pandas as pd

data1 = pd.read_csv('seoul_air_checkData.csv' , encoding='euc-kr')
data2 = pd.read_csv('seoul_air_checkLocation.csv', encoding='euc-kr')
data3 = pd.read_csv('seoul_air_hour_2022.csv')

# print(data1.head())
# print(data2.head())
#print(data3.head())


# print(data1.info())
# print(data2.info())
# print(data3.info())


merge_data = pd.merge(data2,data3 , on='측정소 코드')


mdata = pd.merge(merge_data , data1 , left_on='측정항목' , right_on='측정항목 코드')




cond = (mdata['측정항목 코드'] == 9) & (mdata['평균값'] >=0)

mdata = mdata[cond]



#mdata = mdata[['측정일시' ,'측정소 코드','측정항목','평균값','측정기 상태','국가 기준초과 구분','지자체 기준초과 구분','측정소 이름']]
mdata = mdata[['측정일시' ,'평균값','측정소 이름']]

print(mdata.head())
print(len(mdata[mdata['평균값'] < 10]))
print(len(mdata[(mdata['평균값'] > 10 )& (mdata['평균값'] < 20)]))
print(len(mdata[(mdata['평균값'] > 20 )& (mdata['평균값'] < 30)]))
import seaborn as sns

#target = mdata.pop('평균값')





mdata['측정일시'] = pd.to_datetime(mdata['측정일시'], format='%Y%m%d%H')

mdata['년도'] = mdata['측정일시'].dt.year 
mdata['월'] = mdata['측정일시'].dt.month 
mdata['일'] = mdata['측정일시'].dt.day 
mdata['시간'] = mdata['측정일시'].dt.hour
mdata = mdata.drop(columns='측정일시')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
cond = mdata['평균값'] < 200
mdata = mdata[cond]
sns.boxplot(x='월', y='평균값', data=mdata)
plt.title("월별 미세먼지 평균값 분포")
plt.xlabel("월")
plt.ylabel("평균 미세먼지 농도")
plt.grid(True)

plt.show()  # ✅ 이거 꼭 필요함!

import lightgbm as lgb

mdata['측정소_이름'] =  mdata['측정소 이름'].astype('category')
mdata = mdata.drop(columns='측정소 이름')
print(mdata.info())

train = lgb.Dataset(mdata , target , categorical_feature=['측정소_이름'])

params = {
     'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 20,
    'num_leaves': 128,
    'min_data_in_leaf': 2,
    'learning_rate': 0.2,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(params , train , num_boost_round=100)

temp = pd.DataFrame({
    '년도': [2022,2022,2022,2022,2022,2022,2022,2022,2022,2022,2022,2022],
    '월': [1,2,3,4,5,6,7,8,9,10,11,12],
    '일': [1,1,1,1,1,1,1,1,1,1,1,1],
    '시간': [12,12,12,12,12,12,12,12,12,12,12,12],
    '측정소_이름': ['강남구','강남구','강남구','강남구','강남구','강남구','강남구','강남구','강남구','강남구','강남구','강남구']

})


temp['측정소_이름'] = temp['측정소_이름'].astype('category')
result = model.predict(temp)

print(result)