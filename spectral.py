import numpy as np

def laplacian(W):
    D = np.diag(np.sum(W,1))
    D_invsqrt = np.sqrt(np.linalg.inv(D))

    return np.dot(np.dot(D_invsqrt,(D-W)),D_invsqrt)

def diffusion_operator():
    D = np.diag(np.sum(W,1))
    D_invsqrt = np.sqrt(np.linalg.inv(D))
    T = np.dot(np.dot(D_invsqrt,W),D_invsqrt) # normalized operator, or...

