#使用PyTorch中nn这个库来构建网络
import torch
import torch.nn as nn

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 64
D_in = 1000
H = 100
D_out = 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.

model = nn.Sequential(nn.linear(D_in, H),
                      nn.Relu(),
                      nn.Linear(H, D_out))


# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = nn.MSELoss(reduce = 'sum')


learning_rate = 1e-4


for i in range (500):
    y_pred = model(x)
    loss = loss_fn(y, y_pred)
    loss.backward()
    













