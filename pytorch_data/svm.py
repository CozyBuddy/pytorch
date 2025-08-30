from sklearn import svm, metrics ,datasets ,model_selection
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

iris = datasets.load_iris()
X_train, X_test ,y_train, y_val = model_selection.train_test_split(iris.data, iris.target , test_size=0.6 , random_state=0)

svm = svm.SVC(kernel='linear' , C=1.0 , gamma=0.5)
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)
score = metrics.accuracy_score(y_val,predictions)

print('{0:f}'.format(score))
