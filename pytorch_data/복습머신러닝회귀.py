import pandas as pd

train = pd.read_csv('train_6.csv')
test = pd.read_csv('test_6.csv')




lentrain = len(train)
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mode()[0])
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])

test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mode()[0])
test['Outlet_Size'] = test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])
y_train = train.pop('Item_Outlet_Sales')
data = pd.concat([train,test])

data_O = pd.get_dummies(data)

train = data_O.iloc[:lentrain]
test = data_O.iloc[lentrain:]

print(train.shape)

print(test.shape)


from sklearn.model_selection import train_test_split

X_train ,X_val , y_train, y_val = train_test_split(train,y_train, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train , y_train)

y_pred = rf.predict(X_val)

print(y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val,y_pred)

print('mse ' , mse)

pred = rf.predict(test)

print(pred)

submit = pd.DataFrame({
    'pred' : pred
})

submit.to_csv('result.csv', index=False)