import numpy as np
from scipy import linalg
from scipy.linalg import solve_triangular
from scipy.linalg import tril
from scipy.linalg import triu
from scipy.linalg import norm
from scipy.linalg import blas
from scipy.linalg.blas import dsyrk

def CQRRPT(m, n, M, d_factor, tol):
    # Naive rank estimation threshold
    eps_initial_rank_estimation = 2 * (np.finfo(float).eps ** 0.95)
    eps_mach = np.finfo(float).eps
    # Embedding dimension
    d = int(d_factor * n)
    # Generate a Gaussian Random sketching operator
    S = np.random.normal(0, 1, size=(d, m))
    # Find a sketch
    M_sk = np.matmul(S, M)
    # Perform QRCP on a sketch
    Q_sk, R_sk, J = linalg.qr(M_sk, mode='economic', pivoting=True)
    # Preliminary (naive) rank estimation
    k = n 
    for i in range(n):
        if abs((R_sk[i, i]) / abs(R_sk[0, 0])) < eps_initial_rank_estimation:
            k = i
            break
    # Drop the unnecessary rows and columns from R_sk
    A_sk_k = R_sk[:k, :k]
    # Pivot columns of A
    M_k = M[:, J[:k]]
    # M_pre = M[:, J[1:k]] * (R_sk[1:k, 1:k])^-1
    # There is no option of doing XA=B in scipy, so need to play with transposes.
    M_pre = solve_triangular(A_sk_k.T, M_k.T, lower=True)
    M_pre = M_pre.T
    # Perform Cholesky QR
    L = dsyrk(1.0, M_pre, trans=True)
    L = np.linalg.cholesky(L + triu(L, k=1).T)
    # Re-estimate rank after we have the L-factor form Cholesky QR.
    # The strategy here is the same as in naive rank estimation.
    # This also automatically takes care of any potentical failures in Cholesky factorization.
    # Note that the diagonal of L may not be sorted, so we need to keep the running max/min
    # We expect the loss in the orthogonality of Q to be approximately equal to u * cond(L)^2,
    # where u is the unit roundoff for the numerical type used.
    new_rank = k
    running_max = L[0, 0]
    running_min = L[-1, -1]
    
    
    for i in range(k):
        curr_entry = abs(L[i, i])
        running_max = max(running_max, curr_entry)
        running_min = min(running_min, curr_entry)
        if(running_max / running_min >= np.sqrt(tol / eps_mach)):
            new_rank = i - 1
            break

    k = new_rank
    # There is no option of doing XA=B in scipy, so need to play with transposes.
    Q = solve_triangular(L, M_pre.T, lower=True)
    Q = Q.T
    # Find R
    R = np.matmul(L.T, R_sk[:k, :])

    return Q, R, J



m = 100
n = 50
d_factor = 1.25
tol = np.finfo(float).eps ** 0.85

#M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
M = np.random.normal(0, 100, size=(m, int(n / 2)))
M = np.concatenate((M, M), axis=1)

[Q, R, J] = CQRRPT(m, n, M, d_factor, tol)    

print('||M[:, J] - QR||_F / ||M||_F:', norm(M[:, J] - np.matmul(Q, R)) / norm(M))
