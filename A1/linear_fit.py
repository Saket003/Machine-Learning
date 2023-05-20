import numpy as np
import math

def linear_fit_maxit_noval(x,y,w,N,alpha,it):
    MSE = []
    while(it > 0):
        it = it - 1
        diff = np.dot(x,w) - y

        mse = np.sum(diff**2)/N
        MSE.append(mse)

        grad = 0
        for i in range(N):
            grad = grad + diff[i]*(x[i])
        w = w - (alpha/N)*grad
    return w, MSE[-1]

def linear_fit_maxit(x,y,xval,yval,w,N,alpha,it):
    MSE = []
    MAE = []
    MSE_val = []
    MAE_val = []
    while(it > 0):
        it = it - 1
        diff = np.dot(x,w) - y
        diff_val = np.dot(xval,w) - yval

        mse = np.sum(diff**2)/N
        mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
        mse_val = np.sum(diff_val**2)/N
        mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N
        MSE.append(mse)
        MAE.append(mae)
        MSE_val.append(mse_val)
        MAE_val.append(mae_val)

        grad = 0
        for i in range(N):
            grad = grad + diff[i]*(x[i])
        w = w - (alpha/N)*grad
    

    diff = np.dot(x,w) - y
    diff_val = np.dot(xval,w) - yval

    mse = np.sum(diff**2)/N
    mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
    mse_val = np.sum(diff_val**2)/N
    mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N
    MSE.append(mse)
    MAE.append(mae)
    MSE_val.append(mse_val)
    MAE_val.append(mae_val)

    return w, MSE, MAE, MSE_val, MAE_val


def linear_fit_reltol(x,y,xval,yval,w,N,alpha,bound):
    MSE = []
    MAE = []
    MSE_val = []
    MAE_val = []
    while(True):
        diff = np.dot(x,w) - y
        diff_val = np.dot(xval,w) - yval

        mse = np.sum(diff**2)/N
        mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
        mse_val = np.sum(diff_val**2)/N
        mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N

        if len(MSE_val)>0 and (abs(MSE_val[-1]-mse_val)/mse_val)<bound:
            break
        
        MSE.append(mse)
        MAE.append(mae)
        MSE_val.append(mse_val)
        MAE_val.append(mae_val)

        grad = 0
        for i in range(N):
            grad = grad + diff[i]*(x[i])
        w = w - (alpha/N)*grad
    
    diff = np.dot(x,w) - y
    diff_val = np.dot(xval,w) - yval

    mse = np.sum(diff**2)/N
    mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
    mse_val = np.sum(diff_val**2)/N
    mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N
    MSE.append(mse)
    MAE.append(mae)
    MSE_val.append(mse_val)
    MAE_val.append(mae_val)

    return w, MSE, MAE, MSE_val,MAE_val

def ridge_fit_maxit(x,y,xval,yval,w,N,alpha,it,l):
    MSE = []
    MAE = []
    MSE_val = []
    MAE_val = []

    while(it > 0):
        it = it - 1
        diff = np.dot(x,w) - y
        diff_val = np.dot(xval,w) - yval

        mse = np.sum(diff**2)/N
        mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
        mse_val = np.sum(diff_val**2)/N
        mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N
        MSE.append(mse)
        MAE.append(mae)
        MSE_val.append(mse_val)
        MAE_val.append(mae_val)

        grad = 0
        for i in range(N):
            grad = grad + diff[i]*(x[i])
        grad = (1/N)*grad + (l/N)*w
        w = w - 2*alpha*grad

    diff = np.dot(x,w) - y
    diff_val = np.dot(xval,w) - yval

    mse = np.sum(diff**2)/N
    mae = (np.sum(diff, where = diff > 0) - np.sum(diff, where= diff < 0))/N
    mse_val = np.sum(diff_val**2)/N
    mae_val = (np.sum(diff_val, where = diff_val > 0) - np.sum(diff_val, where= diff_val < 0))/N
    MSE.append(mse)
    MAE.append(mae)
    MSE_val.append(mse_val)
    MAE_val.append(mae_val)
    
    return w, MSE, MAE, MSE_val, MAE_val