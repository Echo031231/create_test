import torch

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')

'''
if(torch.cuda.is_available()):
    print('gpu')
else:
    print('cpu')
'''

N = 64
D_in = 1000
D_out = 10
H = 100

# Creat random input and output data
x = torch.randn(N, D_in, device=device, dtype = dtype)#------(64,1000)
y = torch.randn(N, D_out, device=device, dtype = dtype)#------(64,10)

# Initialize weights
w1 = torch.randn(D_in, H, device=device, dtype = dtype, requires_grad = True)#------(1000,100)
w2 = torch.randn(H,D_out, device=device, dtype = dtype, requires_grad = True)#------(100,10)

'''
h = x * w1 + b1
h_relu = max(0, h)
y_pred = h_relu * w2
'''

learning_rate = 1e-6

for i in range(500):

    # Forward pass: compute y_pred
    h = x.mm(w1) #-----(64,100)
    h_relu = h.clamp(min=0) #-----(64,100)
    y_pred = torch.mm(h_relu, w2) #-----(64,10)

    # Comput and print loss
    loss = (y_pred - y).pow(2).sum()#item放里面就不是一张计算图了
    print(i, loss.item())

    # Backprop to compute gradient of w1 and w2
    
    '''
    # loss to w2 gradient
    grad_y_pred = 2 * (y_pred - y) #-----(64,10)
    grad_w2 = h_relu.t().mm(grad_y_pred)#-----(100,10)

    # loss to w1 gradient
    grad_h_relu = grad_y_pred.mm(w2.transpose(0, 1))#-----(64,100)
    grad_h = grad_h_relu.clone()#-----(64,100)
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h) #------(1000,100)
    '''
    loss.backward()

    # Update weigths with gradients descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
    
    # 梯度清零操作
    w1.grad.zero_()
    w2.grad.zero_()



