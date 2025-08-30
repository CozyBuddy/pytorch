import pandas as pd
import torch 
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('car_evaluation.csv')

print(df.head())

fig_size = plt.rcParams['figure.figsize']
fig_size[0] =8
fig_size[1] = 6

plt.rcParams['figure.figsize'] = fig_size

df.output.value_counts().plot(kind='pie' , autopct='%0.05f%%')

#plt.show()

categorical_columns = ['price' , 'maint' , 'doors' ,'persons' , 'lug_capacity' , 'safety']

for category in categorical_columns:
    df[category] = df[category].astype('category')

price = df['price'].cat.codes.values
maint = df['maint'].cat.codes.values
doors = df['doors'].cat.codes.values
persons = df['persons'].cat.codes.values
lug_capacity = df['lug_capacity'].cat.codes.values
safety = df['safety'].cat.codes.values

categorical_data = np.stack([price,maint,doors,persons,lug_capacity,safety],1)

print(categorical_data[:5])

categorical_data = torch.tensor(categorical_data , dtype=torch.int64)

outputs = pd.get_dummies(df.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

print(categorical_data.shape)
print(outputs.shape)

categorical_column_sizes = [len(df[column].cat.categories) for column in categorical_columns]
print(categorical_column_sizes)
categorical_embedding_sizes = [(col_size , min(50,(col_size+1)//2)) for col_size in categorical_column_sizes]
print(categorical_embedding_sizes)

total_records = 1728
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records: total_records]

train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records -test_records:total_records]

# from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# import torch

# categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']

# # 입력
# ordinal_encoder = OrdinalEncoder()
# categorical_data = ordinal_encoder.fit_transform(df[categorical_columns])

# # 출력
# onehot_encoder = OneHotEncoder(sparse_output=False)
# outputs = onehot_encoder.fit_transform(df[['output']])

# # 텐서 변환
# categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
# outputs = torch.tensor(outputs, dtype=torch.int64).flatten()
# print(categorical_data.shape)
# print(outputs.shape)
# # Embedding info
# categorical_column_sizes = [len(c) for c in ordinal_encoder.categories_]
# categorical_embedding_sizes = [(c, min(50, (c + 1) // 2)) for c in categorical_column_sizes]
# print(categorical_column_sizes)
# print(categorical_embedding_sizes)

# # Train/test
# total_records = 1728
# test_records = int(total_records * 0.2)

# categorical_train_data = categorical_data[:total_records - test_records]
# categorical_test_data = categorical_data[total_records - test_records:]
# train_outputs = outputs[:total_records - test_records]
# test_outputs = outputs[total_records - test_records:]

# print(len(test_outputs))
# print(categorical_train_data.shape)
class Model(nn.Module):
    def __init__(self, embedding_size , output_size, layers ,p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        
        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols
        
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size =i
            
        all_layers.append(nn.Linear(layers[-1] , output_size))
        
        self.layers = nn.Sequential(*all_layers)
        
    def forward(self ,x_categorical):
        embeddings = []
        for i ,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x=torch.cat(embeddings,1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        
        return x
        
        
        
model = Model(categorical_embedding_sizes ,4 ,[200,100,50] ,p=0.4)
print(model)
    
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
epochs = 500
aggregated_losses = []

train_outputs = train_outputs.to(device=device , dtype=torch.int64)

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data).to(device)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)
    
    if i%25 ==1:
        print(f'epoch: { i: 3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()
    
    
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')    

test_outputs = test_outputs.to(device=device , dtype = torch.int64)

print(test_outputs.shape)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val , test_outputs)
    
print(f'loss : {loss :.8f}')

y_val = np.argmax(y_val , axis=1)

print(y_val[:5])

from sklearn.metrics import classification_report ,confusion_matrix , accuracy_score
print(confusion_matrix(test_outputs, y_val))
print(classification_report(test_outputs, y_val))
print(accuracy_score(test_outputs, y_val))