import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)        # 2.7.1+cpu? or 2.7.1+cu121?
print(torch.version.cuda)       # 12.1 ? or None ?
print(torch.cuda.is_available()) # True ? or False ?

train_dataset = torchvision.datasets.FashionMNIST("C:\pytorch\pytorch_data\data" , download=True , transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST("C:\pytorch\pytorch_data\data" , download=True ,train=False, transform=transforms.Compose([transforms.ToTensor()]))


train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=100 ,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=100 ,num_workers=0)

labels_map = { 0:'T-Shirt' , 1 : 'Trouser' ,2: 'Pullover' , 3: 'Dress' , 4: 'Coat' ,5:'Sandal' ,6 :'Shirt' , 7:'Sneaker' , 8:'Bag' , 9 : 'Ankel Boot'}

fig = plt.figure(figsize=(8,8))
columns =4 
rows = 5
for i in range( 1, columns*rows +1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0 , : ,:]
    fig.add_subplot(rows , columns , i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img ,cmap='gray')
    
#plt.show()

class FashionDNN(nn.Module):
    def __init__(self):
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784 , out_features=256)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256 , out_features=128)
        self.fc3 = nn.Linear(in_features=128 , out_features=10)
    
    def forward(self, input_data):
        out = input_data.view(-1,784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

learning_rate = 0.001
model = FashionDNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

print(model)

num_epochs = 5
count = 0

loss_list = []
iteration_list = []
accuracy_list =[]

predictions_list = []
labels_list = []

# for epoch in range(num_epochs):
#     for images , labels in train_loader:
#         images, labels = images.to(device) , labels.to(device)
        
#         train = Variable(images.view(100,1,28,28))
#         labels = Variable(labels)
        
#         outputs = model(train)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         count+=1
        
#         if not (count % 50):
#             total =0 
#             correct =0
#             for images , labels in test_loader:
#                 images ,labels = images.to(device) ,labels.to(device)
#                 labels_list.append(labels)
#                 test = Variable(images.view(100,1,28,28))
#                 outputs = model(test)
#                 pred = torch.max(outputs ,1)[1].to(device)
#                 predictions_list.append(pred)
#                 correct += (pred == labels).sum()
#                 total += len(labels)
                
            
#             accuracy = correct * 100 / total
#             loss_list.append(loss.data)
#             iteration_list.append(count)
#             accuracy_list.append(accuracy)
            
#         if not (count % 500):
#             print('Iteration : {} , Loss : {} , Accuracy : {}%'.format(count ,loss.data, accuracy))
    
    
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        #train = images.view(-1, 1, 28, 28)  # Variable 불필요
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        #print("images.device:", images.device)
        if not (count % 50):
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                test = images.view(-1, 1, 28, 28)
                outputs = model(test).to(device)
                _, pred = torch.max(outputs, 1)
                predictions_list.append(pred)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

            accuracy = correct * 100 / total
            loss_list.append(loss.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print(f"Iteration : {count}, Loss : {loss.item()}, Accuracy : {accuracy}%")
