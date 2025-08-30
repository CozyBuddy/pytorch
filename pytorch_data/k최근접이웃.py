import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

names = ['sepal-length' ,'sepal-width','petal-length' ,'petal-width','Class']

df = pd.read_csv('iris.data')

print(df)

X = df.iloc[: , :-1].values
y = df.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_val = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler

s = StandardScaler()
s.fit(X_train)
X_train = s.transform(X_train)
X_test = s.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_test)

print(accuracy_score(y_val, y_pred))


k =10
acc_array = np.zeros(k)

for k in np.arange(1, k+1 ,1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_val,y_pred)
    acc_array[k-1] = acc
    
max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print('정확도' , max_acc , '최적의 k' , k+1)


