import numpy as np
import math
import linear_fit as lf
import inpu as ip

def main():
    N,x,y,xval,yval = ip.input()

    w = np.zeros(2049)
    alpha = 0.001
    l = 5
    it = 1000
    print("Termination of ridge regression by maxit with lambda = ",l)
    w,MSE,MAE,MSE_val,MAE_val = lf.ridge_fit_maxit(x,y,xval,yval,w,N,alpha,it,l)
    print("w = ",w)
    print("Final MSE and MAE on train are:",MSE[-1]," and ",MAE[-1])
    print("Final MSE and MAE on validation are:",MSE_val[-1]," and ",MAE_val[-1],"\n")

    w = np.zeros(2049)
    alpha = 0.001
    l = 25
    it = 1000
    print("Termination of ridge regression by maxit with lambda = ",l)
    w,MSE,MAE,MSE_val,MAE_val = lf.ridge_fit_maxit(x,y,xval,yval,w,N,alpha,it,l)
    print("w = ",w)
    print("Final MSE and MAE on train are:",MSE[-1]," and ",MAE[-1])
    print("Final MSE and MAE on validation are:",MSE_val[-1]," and ",MAE_val[-1])

if __name__ == "__main__":
    main()