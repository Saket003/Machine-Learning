import numpy as np
import math
import linear_fit as lf
import inpu as ip
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, r_regression
from sklearn.linear_model import Ridge

def main():
    N,x,y,xval,yval = ip.input()

    best_k = SelectKBest(score_func=f_regression, k=10)
    x_new = best_k.fit_transform(x,y)
    xval_new = best_k.transform(xval)
    print("Termination by maxit with 1000 iterations.")
    alpha = 0.1
    bound = 0.000001
    w = np.zeros(10)

    w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_reltol(x_new,y,xval_new,yval,w,N,alpha,bound)
    print("w = ",w)
    print("Final MSE and MAE on train are:",MSE[-1]," and ",MAE[-1])
    print("Final MSE and MAE on validation are:",MSE_val[-1]," and ",MAE_val[-1])

    best_from_model = SelectFromModel(estimator=Ridge(),max_features=10)
    x_new = best_from_model.fit_transform(x,y)
    xval_new = best_from_model.transform(xval)
    print("Termination by maxit with 1000 iterations.")
    w = np.zeros(10)

    w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_reltol(x_new,y,xval_new,yval,w,N,alpha,bound)
    print("w = ",w)
    print("Final MSE and MAE on train are:",MSE[-1]," and ",MAE[-1])
    print("Final MSE and MAE on validation are:",MSE_val[-1]," and ",MAE_val[-1])

if __name__ == "__main__":
    main()