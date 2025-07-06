import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X = pd.read_csv('credit card.csv')

X = X.drop('CUST_ID' , axis=1)
X.fillna(method='ffill' , inplace=True)
print(X.head())

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)

X_principal.columns = ['P1' , 'P2']

print(X_principal.head())

db_default = DBSCAN(eps=0.0375 , min_samples=50).fit(X_principal)
labels = db_default.labels_

colours = {}
colours[0] = 'y'
colours[1] ='g'
colours[2] = 'b'
colours[3] = 'c'
colours[4] ='y'
colours[5] = 'm'
colours[-1] = 'k'

cvec = [colours[label] for label in labels]

r = plt.scatter(X_principal['P1'] , X_principal['P2'] , color='y')
g= plt.scatter(X_principal['P1'] , X_principal['P2'] , color='g')
b = plt.scatter(X_principal['P1'] , X_principal['P2'] , color='b')

c= plt.scatter(X_principal['P1'] , X_principal['P2'] , color='c')
y = plt.scatter(X_principal['P1'] , X_principal['P2'] , color='y')
m = plt.scatter(X_principal['P1'] , X_principal['P2'] , color='m')

k = plt.scatter(X_principal['P1'] , X_principal['P2'] , color='k')



plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'] , X_principal['P2'] , c=cvec)

plt.legend((r,g,b,k) , ('Label 0' , 'Label 1' , 'Label 2' , 'Label -1'))
plt.show()
