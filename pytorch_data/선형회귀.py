import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('weather.csv')

print(df.head())
df.plot(x='MinTemp' , y='MaxTemp' , style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')

plt.show()

x = df['MinTemp'].values.reshape(-1,1)
y = df['MaxTemp'].values.reshape(-1,1)

X_train,X_val , y_train, y_val = train_test_split(x,y,test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_val)

df = pd.DataFrame({'Actual' : y_val.flatten() ,'Predicted' : y_pred.flatten()})

print(df)

plt.scatter(X_val ,y_val , color='gray')
plt.plot(X_val , y_pred ,color='red' , linewidth=2)
plt.show()