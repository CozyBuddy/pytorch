import pandas as pd

y_true = pd.DataFrame(
    [1,1,1,0,0,1,1,1,1,0]
)
y_pred = pd.DataFrame(
    [1,0,1,1,0,0,0,1,1,0]
)

y_true_str = pd.DataFrame([
    'A','A','A','B','B','A','A','A','A','B'
])

y_pred_str = pd.DataFrame([
    'A','B','A','A','B','B','B','A','A','B']
)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true,y_pred)
print(acc)

acc2 = accuracy_score(y_true_str, y_pred_str)
print(acc2)

from sklearn.metrics import precision_score

precision = precision_score(y_true,y_pred)
print(precision)

precision = precision_score(y_true_str , y_pred_str , pos_label='A')

print(precision)

from sklearn.metrics import recall_score

recall = recall_score(y_true,y_pred)
print(recall)

recall = recall_score(y_true_str, y_pred_str, pos_label='A')

print(recall)

from sklearn.metrics import f1_score

f1 = f1_score(y_true,y_pred)
print(f1)

f1 = f1_score(y_true_str, y_pred_str , pos_label='A')
print(f1)

from sklearn.metrics import roc_auc_score
y_true = pd.DataFrame([0,1,0,1,1,0,0,0,1,1])
y_pred_proba = pd.DataFrame([0.4,0.9,0.1,0.3,0.8,0.6,0.4,0.2,0.7,0.6])

roc_auc = roc_auc_score(y_true,y_pred_proba)

print(roc_auc)

y_true_str = pd.DataFrame(['A','B','A','B','B','A','A','A','B','B'])

y_pred_proba_str = pd.DataFrame([0.4,0.9,0.1,0.3,0.8,0.6,0.4,0.2,0.7,0.6])

roc_auc = roc_auc_score(y_true_str,  y_pred_proba_str)
print(roc_auc)

y_true = pd.DataFrame([1,2,3,3,2,1,3,3,2,1])
y_pred = pd.DataFrame([1,2,1,3,2,1,1,2,2,1])

y_true_str = pd.DataFrame(['A','B','C','C','B','A','C','C','B','A'])
y_pred_str = pd.DataFrame(['A','B','A','C','B','A','A','B','B','A'])

from sklearn.metrics import accuracy_score
accu = accuracy_score(y_true,y_pred)

print(accu)

accu = accuracy_score(y_true_str ,y_pred_str)
print(accu)

from sklearn.metrics import precision_score

precision = precision_score(y_true,y_pred,average='macro')
print(precision)

precision = precision_score(y_true_str, y_pred_str, average='macro')

print(precision)

from sklearn.metrics import recall_score

recall = recall_score(y_true,y_pred,average='macro')

print(recall)

recall = recall_score(y_true_str,y_pred_str, average='macro')

print(recall)

from sklearn.metrics import f1_score

f1 = f1_score(y_true,y_pred,average='macro')

print(f1)

f1 = f1_score(y_true_str, y_pred_str ,average='macro')

print(f1)

import pandas as pd

y_true = pd.DataFrame([1,2,5,2,4,4,7,9])
y_pred = pd.DataFrame([1.14,2.53,4.87,3.08,4.21,5.53,7.51,10.32])


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true,y_pred)
print("MSE" , mse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true,y_pred)

print('MAE' , mae)

from sklearn.metrics import r2_score
r2 = r2_score(y_true,y_pred)

print('r2' , r2)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_true,y_pred)

print('rmse', rmse)

from sklearn.metrics import mean_squared_log_error

msle = mean_squared_log_error(y_true,y_pred)

print('msle' , msle)

from sklearn.metrics import root_mean_squared_log_error

rmsle = root_mean_squared_log_error(y_true,y_pred)

print('rmsle' , rmsle)
