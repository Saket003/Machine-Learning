import numpy as np

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    assert X.ndim == 2
    kernel_matrix = X @ X.T
    return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    #Get input parameters - zeta, gamma, Q
    assert X.ndim == 2
    linear_kernel = X @ X.T
    first_order_kernel = np.dot(float(kwargs["gamma"]),linear_kernel) + float(kwargs["zeta"])
    kernel_matrix = np.power(first_order_kernel,float(kwargs["Q"]))
    return kernel_matrix

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    #Get input parameters - gamma
    assert X.ndim == 2
    n = np.size(X,0)
    Y = np.zeros((n,n))
    for i in range(n):
        for j in range (i):
            Y[i][j] = -1*kwargs["gamma"]*np.linalg.norm(X[i]-X[j],ord=2)
    kernel_matrix = np.power(np.exp(1),np.add(Y,Y.T))
    return kernel_matrix

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    #Get input parameters - gamma,r
    assert X.ndim == 2
    linear_kernel = X @ X.T
    scaled_matrix = np.dot(kwargs["gamma"],linear_kernel) + kwargs["r"]
    kernel_matrix = np.tanh(scaled_matrix)
    return kernel_matrix

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    #Get gamma
    assert X.ndim == 2
    n = np.size(X,0)
    Y = np.zeros((n,n))
    for i in range(n):
        for j in range (i):
            Y[i][j] = -1*kwargs["gamma"]*np.linalg.norm(X[i]-X[j],ord=1)
    kernel_matrix = np.power(np.exp(1),np.add(Y,Y.T))
    return kernel_matrix