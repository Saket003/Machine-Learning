import numpy as np
import math
import linear_fit as lf
import inpu as ip

def multi_class_fit(x,y,xval,yval,W,N,alpha,it):
    grad = []
    for i in range (0,8):
        grad.append(np.zeros(W[0].size))
    
    MSE = []
    MSE_val = []
    while(it > 0):
        it = it - 1
        #diff = np.dot(x,w) - y
        #diff_val = np.dot(xval,w) - yval

        #mse = np.sum(diff**2)/N #CC if correct format
        #mse_val = np.sum(diff_val**2)/N
        #MSE.append(mse)
        #MSE_val.append(mse_val)
        
        grad = []
        for i in range (0,8):
            grad.append(np.zeros(W[0].size))
        
        for j in range(N):
            Dr = 1
            for i in range(0,8):
                Dr = Dr + np.exp(np.dot(x[j],W[i]))
            for i in range(0,8):
                y_bool = 1 if y[j] == i+1 else 0
                grad[i] = grad[i] + ((np.exp(np.dot(x[j],W[i]))/Dr)-y_bool)*(x[j])
        
        for i in range(0,8):
            W[i] = W[i] - alpha*grad[i]
    
    return W, MSE, MSE_val


def main():
    N,x,y,xval,yval = ip.input()
    W = []

    for i in range (0,8):
        W.append(np.zeros(2049))

    alpha = 0.01
    it = 1000
    W, MSE, MSE_val = multi_class_fit(x,y,xval,yval,W,N,alpha,it)

    MSE = 0
    for i in range(len(y)):
        Dr = 1
        for j in range(0,8):
            Dr = Dr + np.exp(np.dot(x[i],W[j]))
        
        prob = [0,0,0,0,0,0,0,0,0]
        for j in range(8):
            prob[j] = np.exp(np.dot(x[i],W[j]))/Dr
        prob[8] = 1 - prob[0] - prob[1] - prob[2] - prob[3] - prob[4] - prob[5] - prob[6] - prob[7]

        max_index = prob.index(max(prob))
        MSE = MSE + (y[i] - max_index - 1)*(y[i] - max_index - 1)

    MSE_val = 0
    for i in range(len(yval)):
        Dr = 1
        for j in range(0,8):
            Dr = Dr + np.exp(np.dot(xval[i],W[j]))
        
        prob = [0,0,0,0,0,0,0,0,0]
        for j in range(8):
            prob[j] = np.exp(np.dot(xval[i],W[j]))/Dr
        prob[8] = 1 - prob[0] - prob[1] - prob[2] - prob[3] - prob[4] - prob[5] - prob[6] - prob[7]

        max_index = prob.index(max(prob))
        MSE_val = MSE_val + (yval[i] - max_index - 1)*(yval[i] - max_index - 1)
    
    MSE = MSE/len(y)
    MSE_val = MSE_val/len(yval)
    print("Final MSE on train is:",MSE)
    print("Final MSE on validation is:",MSE_val)
    
    

if __name__ == "__main__":
    main()