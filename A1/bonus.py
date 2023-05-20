import numpy as np
import math
import inpu as ip

def logistic_single(x,y,w,N,alpha,num):
    MSE = []
    MSE_val = []
    it = 1000 #Check better terminating

    y_bool = []
    for t in y:
        y_bool.append(1 if t==num else 0)

    while(it > 0):
        it = it - 1

        grad = 0
        for i in range(N):
            grad = grad + (sigmoid(x[i],w) - y_bool[i])*(x[i])
        grad = grad/N
        w = w - alpha*grad

    return w, MSE, MSE_val


def sigmoid(x,w):
    return 1/(1+np.exp(-np.dot(x,w)))

def main():
    N,x,y,xval,yval = ip.input()
    P = np.zeros(9)
    alpha = 0.01

    for i in range(8):
        w = np.zeros()
        w = logistic_single(x,y,w,N,alpha,i+1)
        


if __name__ == "__main__":
    main()