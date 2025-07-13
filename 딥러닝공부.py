import torch
print(torch.tensor([[1,2],[3,4]]))
print(torch.tensor([[1,2],[3,4]] , device='cuda:0'))

temp = torch.tensor([[1,2],[3,4]])

print(temp.numpy())
print(temp.view(4,1))
print(temp.view(-1))

print(temp.view(-1,1))
print(temp.view(1,-1))