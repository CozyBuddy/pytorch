import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor

import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)


import torchvision.transforms as transforms

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) , (1.0 ,))
])

from torchvision.datasets import MNIST

download_root = '../chap07/MNIST_DATASET'

train_dataset = MNIST(download_root , transform=mnist_transform , train=True , download=True)
valid_dataset = MNIST(download_root , transform=mnist_transform , train=False , download=True)
test_dataset = MNIST(download_root , transform=mnist_transform , train=False , download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset , batch_size=batch_size , shuffle=True)
valid_loader = DataLoader(dataset=test_dataset ,batch_size= batch_size ,shuffle=True)
test_loader = DataLoader(dataset=test_dataset , batch_size= batch_size , shuffle=True)

batch_size = 100
n_iters =5000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))


class LSTMCell(nn.Module):
    def __init__(self, input_size ,hidden_size , bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size , 4* hidden_size ,bias =bias)
        self.h2h = nn.Linear(hidden_size ,4 * hidden_size , bias=bias)
        self.reset_parameters()

    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)

    
    def forward(self, x ,hidden):
        hx , cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
        #gates = gates.squeeze()
        ingate , forgetgate , cellgate ,outgate = gates.chunk(4,1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy= torch.mul(cx ,forgetgate) + torch.mul(ingate ,cellgate)
        hy = torch.mul(outgate , F.tanh(cy))
        return (hy,cy)
    

class LSTMModel(nn.Module):
    def __init__(self, input_dim , hidden_dim , layer_dim , output_dim ,bias=True):
        super(LSTMModel , self).__init__()
        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim , hidden_dim , layer_dim)
        self.fc = nn.Linear(hidden_dim , output_dim)

    def forward(self,x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim , x.size(0) ,self.hidden_dim).cuda())
        else :
            h0 = Variable(torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim))

        
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim).cuda())
        
        else :
            c0 = Variable(torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim))


        
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]

        for seq in range(x.size(1)):
            hn , cn = self.lstm(x[:,seq,:] , (hn,cn))

            outs.append(hn)


        out = outs[-1]
        out = self.fc(out)
        return out
    

input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10

model = LSTMModel(input_dim , hidden_dim , layer_dim , output_dim)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)


seq_dim = 28
loss_list = []
iter = 0
for epoch in range(num_epochs):
    for i , (images ,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1,seq_dim , input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1,seq_dim,input_dim))
            labels = Variable(labels)


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs ,labels)

        if torch.cuda.is_available():
            loss.cuda()

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        iter += 1


        if iter % 500 == 0:
            correct = 0
            total = 0
            for images ,labels in valid_loader:
                if torch.cuda.is_available():
                    images = Variable(images.view(-1 , seq_dim , input_dim).cuda())

                else:
                    images = Variable(images.view(-1 , seq_dim ,input_dim))


                outputs =model(images)
                _, predicted = torch.max(outputs.data,1)

                total+= labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()

                else:
                    correct += (predicted == labels).sum()

        
            accuracy = 100 * correct / total
            print('Iteration: {} Loss: {} Accuracy: {}'.format(iter,loss.item(), accuracy))

    


def evaluate(model , val_iter):
    corrects , total ,total_loss = 0, 0, 0
    model.eval()
    for images ,labels in val_iter:
        images = images.view(-1, seq_dim, input_dim).to(device)
        labels = labels.to(device)

        
        logit = model(images).to(device)
        loss = F.cross_entropy(logit , labels , reduction='sum')
        _ , predicted = torch.max(logit.data , 1)
        total += labels.size(0)
        total_loss += loss.item()
        corrects += (predicted == labels).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss , avg_accuracy




test_loss , test_acc = evaluate(model , test_loader)
print('Test Loss: %5.2f | test accuracy: %5.2f' % (test_loss , test_acc)) 