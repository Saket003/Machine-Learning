import numpy as np
import math
import linear_fit as lf
import inpu as ip

def main():
    N,x,y,xval,yval = ip.input()

    for i in range(1,4):
        alpha = math.pow(10,-i)
        print("For alpha = ",alpha,",")

        w = np.zeros(2049)
        it = [40,50,1000]
        print("Termination by maxit with ",it[i-1]," iterations.")
        w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(x,y,xval,yval,w,N,alpha,it[i-1])
        print("w = ",w)
        print("Final MSE and MAE on train are:",MSE[-1]," and ",MAE[-1])
        print("Final MSE and MAE on validation are:",MSE_val[-1]," and ",MAE_val[-1])

        w = np.zeros(2049)
        if(i==1 or i==2):
            print("Since gradient descent is diverging, reltol will not be able to find solution either \n")
            continue
        bound = 0.000001
        print("Termination by reltol with bound = ",bound)
        w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_reltol(x,y,xval,yval,w,N,alpha,bound)
        print("w = ",w)
        print("Final MSE and MAE on train are:",MSE[-1]," and ",MAE[-1])
        print("Final MSE and MAE on validation are:",MSE_val[-1]," and ",MAE_val[-1],"\n")

if __name__ == "__main__":
    main()