import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        
        print(df.head())
        self.x = df.iloc[: ,0].values
        self.y = df.iloc[: ,1].values
        self.length = len(df)
        
    def __getitem__(self,index):
        x = torch.FloatTensor([self.x[index] ** 2 , self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        
        return x, y
    
    def __len__(self):
        return self.length
    
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)
        
    def forward(self,x):
        x = self.layer(x)
        return x
    
    
train_dataset = CustomDataset('non_linear.csv')
train_dataloader = DataLoader(train_dataset, batch_size= 2 , shuffle=True , drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters() , lr=0.0001)



for epoch in range(10000):
    cost = 0.0
    
    for x,y in train_dataloader:
  
        x = x.to(device)
        y= y.to(device)
        
        output = model(x)
        loss = criterion(output,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss
        
    
    cost = cost / len(train_dataloader)
    
    if(epoch +1) %1000 ==0 :
        print(f'Epoch : {epoch+1:4d} , Model:{list(model.parameters())} , Cost : {cost:.3f}')
        
        

with torch.no_grad(): ## 기울기 계산을 비활성화 하는 클래스
    model.eval()
    inputs = torch.FloatTensor(
        [
            [1**2 ,1],
            [5**2,5],
            [11**2 ,11]
        ]
    ).to(device)
    
    outputs = model(inputs)
    print(outputs)
    
torch.save(model , 'model.pt')
torch.save(model.state_dict() , 'model_state_dict.pt')


model_state_dict = torch.load('model_state_dict.pt', map_location=True)
model.load_state_dict(model_state_dict)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([
        [1**2,1],
        [5**2 , 5],
        [11**2 , 11]
    ]).to(device)
    
    outputs = model(inputs)
    

