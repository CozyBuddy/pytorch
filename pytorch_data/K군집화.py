import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('sales data.csv')

print(df)

categorical_features = ['Channel' , 'Region']
continuous_features = ['Fresh' , 'Milk' ,'Grocery' , 'Frozen' , 'Detergents_Papers' , 'Delicassen']

for col in categorical_features:
    dummies = pd.get_dummies(df[col] , prefix=col)
    df = pd.concat([df,dummies] , axis=1)
    df.drop(col , axis=1 , inplace=True)
    
print(df.head())


mms = MinMaxScaler()
mms.fit(df)
data_transformed = mms.transform(df)

sum_of_squared_distances = []

K = range(1,15)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    sum_of_squared_distances.append(km.inertia_)
    
print(sum_of_squared_distances)
plt.plot(K ,sum_of_squared_distances, 'bx-')

plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('Optimal k')
plt.show()

