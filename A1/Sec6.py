import numpy as np
import math
import linear_fit as lf
import inpu as ip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, r_regression


def partc():
    print("c)")
    N,x,y,xval,yval = ip.input()

    for i in range(1,4):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=((4-i)/4))
        w = np.zeros(2049)
        alpha = 0.001
        #bound = 0.0001
        it = 50
        Ntrain = int(N*i/4)
        print("Fraction of training set:",Ntrain/N)
        w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(xtrain,ytrain,xval,yval,w,Ntrain,alpha,it)
        plt.plot(MSE,label = "Training")
        plt.plot(MSE_val,label = "Validation")
        plt.legend()
        plt.show()
        print("Final MSE on train = ",MSE[-1])
        print("Final MSE on validation = ",MSE_val[-1])

    w = np.zeros(2049)
    alpha = 0.001
    it = 50
    print("Fraction of training set: 1")
    w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(x,y,xval,yval,w,N,alpha,it)
    plt.plot(MSE,label = "Training")
    plt.plot(MSE_val,label = "Validation")
    plt.legend()
    plt.show()
    print("Final MSE on train = ",MSE[-1])
    print("Final MSE on validation = ",MSE_val[-1])
    
def partd():
    print("d)")
    N,x,y,xval,yval = ip.input()
    
    xtrain, xtrain2, ytrain, ytrain2 = train_test_split(x, y, test_size=0.5)
    w1= np.zeros(2049)
    w2= np.zeros(2049)
    alpha = 0.001
    it = 50
    w1,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(xtrain,ytrain,xval,yval,w1,int(N/2),alpha,it)
    w2,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(xtrain2,ytrain2,xval,yval,w2,int(N/2),alpha,it)
    
    abs_diff_linear = 0
    for i in range(len(ytrain)):
        abs_diff_linear = abs_diff_linear + abs(np.dot(xtrain[i],w1) - np.dot(xtrain2[i],w2))
    abs_diff_linear = abs_diff_linear/(N/2)
    print("Mean absolute difference = ",abs_diff_linear)

    w1= np.zeros(2049)
    w2= np.zeros(2049)
    alpha = 0.001
    it = 50
    l = 25
    w1,MSE,MAE,MSE_val,MAE_val = lf.ridge_fit_maxit(xtrain,ytrain,xval,yval,w1,int(N/2),alpha,it,l)
    w2,MSE,MAE,MSE_val,MAE_val = lf.ridge_fit_maxit(xtrain2,ytrain2,xval,yval,w2,int(N/2),alpha,it,l)

    abs_diff_ridge = 0
    for i in range(len(ytrain)):
        abs_diff_ridge = abs_diff_ridge + abs(np.dot(xtrain[i],w1) - np.dot(xtrain2[i],w2))
    abs_diff_ridge = abs_diff_ridge/(N/2)
    print("Mean absolute difference = ",abs_diff_ridge)

def parte():
    print("e)")
    N,x,y,xval,yval = ip.input()
    alpha = 0.001
    w = np.zeros(2049)
    it = 50
    w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(x,y,xval,yval,w,N,alpha,it)

    Sum_MSE = [0,0,0,0,0,0,0,0,0]
    Num_Score = [0,0,0,0,0,0,0,0,0]
    Avg_MSE = [0,0,0,0,0,0,0,0,0]

    for i in range(len(y)):
        Sum_MSE[y[i]-1] = Sum_MSE[y[i]-1] + (np.dot(x[i],w) - y[i])*(np.dot(x[i],w) - y[i])
        Num_Score[y[i]-1] = Num_Score[y[i]-1] + 1

    for i in range(9):
        Avg_MSE[i] = Sum_MSE[i]/Num_Score[i]

    z = [1,2,3,4,5,6,7,8,9]
    plt.bar(z,Sum_MSE, label = "Sum")
    plt.bar(z,Avg_MSE, label = "Average")
    plt.xlabel('Score')
    plt.legend()
    plt.show()

def partf():
    print("f)")
    N,x,y,xval,yval = ip.input()
    alpha = 0.001
    w = np.zeros(2049)
    it = 50
    MSE_plot = [0,0,0,0]
    w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_maxit(x,y,xval,yval,w,N,alpha,it)
    MSE_plot[3] = MSE[-1]
    K = [10,100,1000,2049]
    for i in range(3):
        best_k = SelectKBest(score_func=f_regression, k=K[i])
        x_new = best_k.fit_transform(x,y)
        xval_new = best_k.transform(xval)
        alpha = 0.001
        bound = 0.0001
        w = np.zeros(K[i])

        w,MSE,MAE,MSE_val,MAE_val = lf.linear_fit_reltol(x_new,y,xval_new,yval,w,N,alpha,bound)
        MSE_plot[i] = MSE[-1]
    plt.plot(K,MSE_plot)
    plt.xlabel("# of Training Features")
    plt.show()

def main():
    #Part1
    """
    You need to plot the training and validation MSE loss (in the same graph) over the iterations
(x-axis: iterations, y-axis: loss). Find an estimate of the number of iterations required by your
algorithm to converge; divide that by 20 and then store the values of loss after each such interval
to get roughly 20 points, and mention your observations in the report. You need to perform this
for sections 3.1, 3.2, 3.4 and 3.5. Compare them and report your observations.
    """

    #Part 2
    #Normalization
    #Normalize the data by the following equation:
    #Perform section 3.1 using normalized data and plot the training and validation MSE loss over the iterations.

    #Part 3
    #partc()

    #Part 4
    #partd()

    #Part 5
    #parte()

    #Part 6
    #partf()


if __name__ == "__main__":
    main()