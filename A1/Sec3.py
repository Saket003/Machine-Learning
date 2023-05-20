import numpy as np
import math
import linear_fit as lf
import inpu as ip
from sklearn.linear_model import LinearRegression

def main():
    N,x,y,xval,yval = ip.input()

    reg = LinearRegression().fit(x,y)
    w = reg.coef_

    diff = np.dot(x,w) - y
    mse = np.sum(diff**2)/N
    mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
    print("Final MSE and MAE on train are:",mse," and ",mae)

    diff_val = np.dot(xval,w) - yval
    mse_val = np.sum(diff_val**2)/N
    mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N
    print("Final MSE and MAE on validation are:",mse_val," and ",mae_val)
    

if __name__ == "__main__":
    main()