import torch 
from torch import nn
from torch import optim

x = []

for i in range(1,31):
    x.append([i])

x = torch.FloatTensor(x)

print(x)

y = torch.FloatTensor([[0.94],[1.98],[2.88],[3.92],[3.96],[4.55],[5.64],[6.3],[7.44],[9.1],[8.46],[9.5],
                       [10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],[17.46],[19.8],[18],[21.34]
                       ,[22],[22.5],[24.57],[26.04],[21.6],[28.8]])

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if (epoch + 1 ) % 1000 == 0 :
        print(f'Epoch : {epoch+1:4d} , Model: {list(model.parameters())} , Cost: {cost:.3f}')
    
    
    
    
class Dataset:
    def __init__(self, data, *arg, **kwargs):
        self.data = data
    
    def __getitem__(self,index):
        return tuple(t[index] for t in self.data.tensors)
    
    def __len__(self):
        return self.data[0].size(0)