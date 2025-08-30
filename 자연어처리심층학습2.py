import torch 
from torch import optim


x = []

for i in range(1,31):
    x.append([i])

x = torch.FloatTensor(x)

print(x)

y = torch.FloatTensor([[0.94],[1.98],[2.88],[3.92],[3.96],[4.55],[5.64],[6.3],[7.44],[9.1],[8.46],[9.5],
                       [10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],[17.46],[19.8],[18],[21.34]
                       ,[22],[22.5],[24.57],[26.04],[21.6],[28.8]])

weight = torch.zeros(1,requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001

optimizer = optim.SGD([weight, bias], lr=learning_rate)

for epoch in range(10):
    hypothesis = x * weight + bias
    cost = torch.mean((hypothesis -y)**2)

    print(f'Epoch : {epoch+1 : 4d}')
    print(f'Step[1] : Gradient : {weight.grad} , Weight : {weight.item():.5f}')

    optimizer.zero_grad()
    print(f'Step[2] : Gradient : {weight.grad} , Weight : {weight.item():.5f}')
    
    cost.backward()
    print(f'Step[3] : Gradient : {weight.grad} , Weight : {weight.item():.5f}')
    
    optimizer.step()
    print(f'Step[4] : Gradient : {weight.grad} , Weight : {weight.item():.5f}')
    

    if (epoch+1) % 1000 ==0:
        print(f'Epoch : {epoch+1 : 4d} ,Weight : {weight.item(): .3f}, Bias: {bias.item():.3f} , Cost: {cost:.3f}')

from torch import nn

# layer = torch.nn.Linear(
#     in_features,
#     out_features,
#     bias=True,
#     device=None,
#     dtype=None
# )