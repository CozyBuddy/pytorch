import torch
from torch import nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)
        
    def forward(self, x):
        x = self.layer(x)
        return x
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomModel().to(device)
    
model_state_dict = torch.load('model_state_dict.pt', map_location=device)
model.load_state_dict(model_state_dict)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([
        [1**2,1],
        [5**2 , 5],
        [11**2 , 11]
    ]).to(device)
    
    outputs = model(inputs)
    print(outputs)
    

