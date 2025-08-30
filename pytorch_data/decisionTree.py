import pandas as pd

df = pd.read_csv('titanic.csv')

print(df)

df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]

df['Sex'] = df['Sex'].map({'male': 0 , 'female' : 1})

df = df.dropna()

X= df.drop('Survived',axis=1)
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test , y_train ,y_val = train_test_split(X, y , random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_val, y_pred))

from sklearn.metrics import confusion_matrix
print(pd.DataFrame(
    confusion_matrix(y_val,y_pred),
    columns=['Predicte Not Survived' , 'Predicted Survival'] ,
    index = ['True Not Survived' , 'True Survival']
))

from sklearn.datasets import load_digits

digits = load_digits()

print("Image Data Shape" , digits.data.shape)
print("Label Data Shape" , digits.target.shape)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))

for index , (image ,label) in enumerate(zip(digits.data[0:5] , digits.target[0:5])):
    plt.subplot(1,5 ,index+1)
    plt.imshow(np.reshape(image , (8,8)) , cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
    
#plt.show()

from sklearn.model_selection import train_test_split

X_train,X_val , y_train , y_val = train_test_split(digits.data , digits.target , test_size=0.25 ,random_state=0)

from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train,y_train)

print(logisticRegr.predict(X_val[0].reshape(1,-1)))
print(logisticRegr.predict(X_val[0:10]))

pred = logisticRegr.predict(X_val)
score = logisticRegr.score(X_val , y_val)

print(score)

import numpy as np
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_val,pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True ,fmt='.3f' , linewidths=.5 , square=True , cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy_Score : {0}' .format(score)

plt.title(all_sample_title , size=15)
plt.show()