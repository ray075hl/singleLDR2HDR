"""
WLS filter: Edge-preserving smoothing based onthe weightd least squares
optimization framework, as described in Farbman, Fattal, Lischinski, and
Szeliski, "Edge-Preserving Decompositions for Multi-Scale Tone and Detail
Manipulation", ACM Transactions on Graphics, 27(3), August 2008.

Given an input image IN, we seek a new image OUT, which, on the one hand,
is as close as possible to IN, and, at the same time, is as smooth as
possible everywhere, except across significant gradients in L.
"""
import cv2 
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, lsqr

def wlsFilter(IN, Lambda=1.0, Alpha=1.2):
    """
    IN        : Input image (2D grayscale image, type float)
    Lambda    : Balances between the data term and the smoothness term.
                Increasing lbda will produce smoother images.
                Default value is 1.0
    Alpha     : Gives a degree of control over the affinities by 
                non-lineary scaling the gradients. Increasing alpha 
                will result in sharper preserved edges. Default value: 1.2
    """
    
    L = np.log(IN+1e-22)        # Source image for the affinity matrix. log_e(IN)
    smallNum = 1e-6
    height, width = IN.shape
    k = height * width

    # Compute affinities between adjacent pixels based on gradients of L
    dy = np.diff(L, n=1, axis=0)   # axis=0 is vertical direction

    dy = -Lambda/(np.abs(dy)**Alpha + smallNum)
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')    # add zeros row
    dy = dy.flatten(order='F')

    dx = np.diff(L, n=1, axis=1)

    dx = -Lambda/(np.abs(dx)**Alpha + smallNum)
    dx = np.pad(dx, ((0,0),(0,1)), 'constant')    # add zeros col 
    dx = dx.flatten(order='F')
    # Construct a five-point spatially inhomogeneous Laplacian matrix
    
    B = np.concatenate([[dx], [dy]], axis=0)
    d = np.array([-height,  -1])

    A = spdiags(B, d, k, k) 

    e = dx 
    w = np.pad(dx, (height, 0), 'constant'); w = w[0:-height]
    s = dy
    n = np.pad(dy, (1, 0), 'constant'); n = n[0:-1]

    D = 1.0 - (e + w + s + n)

    A = A + A.transpose() + spdiags(D, 0, k, k)

    # Solve
    OUT = spsolve(A, IN.flatten(order='F'))
    return np.reshape(OUT, (height, width), order='F') 


# Unit test
if __name__ == '__main__':
    image = cv2.imread('1.png')
    if image.shape[2] == 4:        # Format RGBA
        image = image[:,:, 0:3]    # Discard alpha channel 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image1 = 1.0*image / np.max(image)
    result = wlsFilter(image1)
    cv2.imshow('1', result)
    cv2.waitKey(0)

