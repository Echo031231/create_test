#使用numpy来搭建两层神经网络
import numpy as np

# 更改测试
import random
import time
import turtle


N = 64
D_in = 1000
D_out = 10
H = 100

# Creat random input and output data
x = np.random.randn(N, D_in) # N维, D_in为每一维的特征数
y = np.random.randn(N, D_out)

#Random initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# x-----(64,1000)
# y-----(64,10)
# w1----(1000,100)
# w2----(100,10)


#学习速率
learning_rate = 1e-6


# h = X*W1 + b1 ------(64,100)
# a = max(0,h) ------(64,100)
# y = a*W2 + b2 ------(64,10)


for i in range (500):
    # Forward pass: compute predicted y
    h = x.dot(w1) # (64,100)
    h_relu = np.maximum(h, 0)#非线性激活---增加神经网络的非线性表达能力 # (64,100)
    y_pred = h_relu.dot(w2) # (64,10)

    #compute and print loss
    loss = np.square(y_pred - y).sum()# (y_pred - y)------(64,10)
    print(i, loss)


    # Backprop to compute gradients of w1 and w2 with respect to loss
    # loss = (y_pred - y) ** 2
    
    #loss对w2的导数:
    grad_y_pred = 2.0 * (y_pred - y)
    #loss对y_pred的偏导grad_y_pred尺寸与y_pred一样----(64,10)  
    #y_pred对w2的偏导为h_relu----(64,100)
    grad_w2 = h_relu.T.dot(grad_y_pred)#grad_w2(100,10)也就是w2的尺寸                             
    
    
    #loss对w1的导数:
    grad_h_relu = grad_y_pred.dot(w2.T)# (64,10)   (100,10).T----(10,100)-------------grad_h_relu--(64,100)
    #创建一个grad_h_relu的副本，并赋值给grad_h--起到改变grad_h元素值不会对grad_h_relu元素值产生影响
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0#这行代码使用布尔索引将 grad_h 数组中 h 小于 0 的元素设置为 0----(64,100)
    grad_w1 = x.T.dot(grad_h)#-----(1000,100)


    # Update weights
    #梯度下降法
    #现在已有loss对w1和w2的梯度，即是loss(在这两个参数上)在此方向上变化最快
    #loss(w)----在梯度的反方向上更新w，找到w值使得loss最小
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2





    




    


























