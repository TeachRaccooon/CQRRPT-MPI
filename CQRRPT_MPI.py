from mpi4py import MPI
import numpy as np
from scipy import linalg
from scipy.linalg import solve_triangular
from scipy.linalg import tril
from scipy.linalg import triu
from scipy.linalg import norm
from scipy.linalg import blas
from scipy.linalg.blas import dsyrk

# Use mpirun -np 'numprocs' python3 CQRRPT_MPI.py to run
def CQRRPT_MPI(comm, m, n, M, d_factor, tol):

    # Processes info
    rank = comm.Get_rank()
    size = comm.Get_size()

    eps_mach = np.finfo(float).eps

    # Naive rank estimation threshold
    eps_initial_rank_estimation = 2 * (eps_mach ** 0.95)
    # Embedding dimension
    d = int(d_factor * n)
    
    if rank == 0:
        # Generate a Gaussian Random sketching operator
        S = np.random.normal(0, 1, size=(d, m))
        
        A = np.matmul(S, M)

        # Split S into sub-arrays along required axis
        arrs = np.split(S, size, axis=1)
        # Flatten the sub-arrays
        raveled = [np.ravel(arr) for arr in arrs]
        # Join them back up into a 1D array
        S = np.concatenate(raveled)
    else:
        S = None

    # Partition the input matrix M by rows
    #Technically, the below vals must be equal
    elems_per_chunk = m // size

    # Local partitions of M and S for each processor
    chunk_M = np.empty((elems_per_chunk, n), dtype=float)
    chunk_S = np.empty((d, elems_per_chunk), dtype=float)
    
    comm.Scatterv(S, chunk_S, root=0)
    comm.Scatter(M, chunk_M, root=0)

    # Find local products of chunks of columns of S with chunks of rows of M
    chunk_M_sk = np.matmul(chunk_S, chunk_M)
    
    # M_sk will be available in full on all processors
    M_sk = np.empty((d, n), dtype=float)
    comm.Allreduce(chunk_M_sk, M_sk, op=MPI.SUM)

    #---------------------BELOW WORK IS EQUIVALENT ON EACH PROCESS---------------------#
    # This is done in order to reduce communication

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
    # Pivot columns of each row of M
    chunk_M_k = chunk_M[:, J[:k]]
    # M_pre = M[:, J[1:k]] * (R_sk[1:k, 1:k])^-1
    # There is no option of doing XA=B in scipy, so need to play with transposes.
    chunk_M_pre = solve_triangular(A_sk_k.T, chunk_M_k.T, lower=True)
    chunk_M_pre = chunk_M_pre.T
    #---------------------ABOVE WORK IS EQUIVALENT ON EACH PROCESS---------------------#
    # Сhunks of M_pre can now be gathered if needed. 
    #print("Chunk M_pre")
    #print(chunk_M_pre)
    
    # Perform Cholesky QR
    # Find M_pre.T * M_pre for each process
    chunk_L = dsyrk(1.0, chunk_M_pre, trans=True)
    # L will be available in full on all processors
    L = np.empty((k, k), dtype=float)
    comm.Allreduce(chunk_L, L, op=MPI.SUM)

    #---------------------BELOW WORK IS EQUIVALENT ON EACH PROCESS---------------------#
    # Find cholesky factorization of the symmetric matrix M_pre.T * M_pre
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
    #---------------------ABOVE WORK IS EQUIVALENT ON EACH PROCESS---------------------#

    # There is no option of doing XA=B in scipy, so need to play with transposes.
    chunk_Q = solve_triangular(L, chunk_M_pre.T, lower=True)
    chunk_Q = chunk_Q.T

    # Only rank 0 will have Q and R
    if rank == 0:
        Q = np.empty((m, k), dtype=float)
        # Find R
        R = np.matmul(L.T, R_sk[:k, :])
    else:
        R = None
        Q = None
    # Get full Q
    comm.Gather(chunk_Q, Q, root=0)

    return Q, R, J

# Initialize MPI
comm = MPI.COMM_WORLD
m = 100
n = 50
d_factor = 1.25
tol = np.finfo(float).eps ** 0.85

#M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
M = np.random.normal(0, 100, size=(m, int(n / 2)))
M = np.concatenate((M, M), axis=1)

[Q, R, J] = CQRRPT_MPI(comm, m, n, M, d_factor, tol)

if comm.Get_rank() == 0:
    print('||M[:, J] - QR||_F / ||M||_F:', norm(M[:, J] - np.matmul(Q, R)) / norm(M))
