# model_train.py
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

# 예제 데이터
df = pd.read_csv('tree.csv')
print(df.info())
target = df.pop('성장률')

train = pd.get_dummies(df)

rf = RandomForestRegressor()

rf.fit(train,target)

# 모델 저장
joblib.dump(rf, 'model.pkl')
