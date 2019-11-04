import time
import numpy as np
from tqdm import trange
import scipy.sparse as sp
from scipy.linalg import norm
from joblib import Parallel, delayed
from vezda.math_utils import humanReadable
from vezda.svd_utils import load_svd, svd_needs_recomputing, compute_svd
from vezda.LinearOperators import asConvolutionalOperator


def scipy_lsmr(A, b, damp, atol, btol):
    return sp.linalg.lsmr(A, b, damp, atol, btol)[0]

def scipy_lsqr(A, b, damp, atol, btol):
    return sp.linalg.lsqr(A, b, damp, atol, btol)[0]

def inverse_svd(V, Sp, Uh, b):
    return V.dot(Sp.dot(Uh.dot(b)))

#==============================================================================
# Super class (Parent class)
# A class for solving linear systems Ax = b
#
# Class data objects: 
#   linear operator: A
#   right-hand side vectors: B = [b1 b2 ... bn]
#
# Class methods:
#   solve by iterative least-squares: solve_lsmr
#   solve by iterative least-squares: solve_lsqr
#   solve by singular-value decomposition: solve_svd
#==============================================================================
class LinearSystem(object):
    
    def __init__(self, LinearOperator, rhs_vectors):
        self.A = LinearOperator
        self.B = rhs_vectors
        
        
    def solve_lsmr(self, damp=0.0, atol=1.0e-8, btol=1.0e-8, nproc=1):
        M, N = self.A.shape
        K = self.B.shape[2]
        
        if nproc != 1:
            startTime = time.time()
            X = Parallel(n_jobs=nproc, verbose=11)(
                    delayed(scipy_lsmr)(self.A, self.B[:, :, i].reshape(M),
                            damp=damp, atol=atol, btol=btol)
                    for i in range(K))
            endTime = time.time()
            X = np.asarray(X, dtype=self.A.dtype).T
        
        else:
            # initialize solution matrix X
            X = np.zeros((N, K), dtype=self.A.dtype)
            
            startTime = time.time()
            for i in trange(K):
                X[:, i] = scipy_lsmr(self.A, self.B[:, :, i].reshape(M),
                 damp=damp, atol=atol, btol=btol)
            endTime = time.time()
            
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        return X
    
    def solve_lsqr(self, damp=0.0, atol=1.0e-8, btol=1.0e-8, nproc=1):
        M, N = self.A.shape
        K = self.B.shape[2]
        
        if nproc != 1:
            startTime = time.time()
            X = Parallel(n_jobs=nproc, verbose=11)(
                    delayed(scipy_lsqr)(self.A, self.B[:, :, i].reshape(M),
                            damp=damp, atol=atol, btol=btol)
                    for i in range(K))
            endTime = time.time()
            X = np.asarray(X, dtype=self.A.dtype).T
        
        else:
            # initialize solution matrix X
            X = np.zeros((N, K), dtype=self.A.dtype)
            
            startTime = time.time()
            for i in trange(K):
                X[:, i] = scipy_lsqr(self.A, self.B[:, :, i].reshape(M),
                 damp=damp, atol=atol, btol=btol)
            endTime = time.time()
            
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        return X
    
    def solve_svd(self, U, s, Vh, alpha=0.0, nproc=1):
        #======================================================================
        # Construct the pseudoinverse of A : A+ = V Sp Uh
        if np.issubdtype(U.dtype, np.complexfloating):
            # singular vectors are complex
            Uh = U.getH()
            V = Vh.getH()
        else:
            # singular vectors are real
            Uh = U.T
            V = Vh.T
        
        # Construct the diagonal matrix 'Sp' from 's'
        s = np.divide(s, alpha + s**2)
        Sp = sp.diags(s)
        #======================================================================
        # Apply SVD to obtain solution matrix X
        M, N = self.A.shape
        K = self.B.shape[2]
        
        if nproc != 1:
            startTime = time.time()
            X = Parallel(n_jobs=nproc, verbose=11)(
                    delayed(inverse_svd)(V, Sp, Uh, self.B[:, :, i].reshape(M))
                    for i in range(K))
            endTime = time.time()
            X = np.asarray(X, dtype=self.A.dtype).T
        
        else:
            # initialize solution matrix X
            X = np.zeros((N, K), dtype=self.A.dtype)
            
            startTime = time.time()
            for i in trange(K):
                X[:, i] = inverse_svd(V, Sp, Uh, self.B[:, :, i].reshape(M))
            endTime = time.time()
        
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        return X


#==============================================================================
# Subclass (Derived class)
# A class for solving linear sampling problems of the form Ax = b
#
# Class data objects: 
#   kernel: data or test functions
#   right-hand side vectors: B = [b1 b2 ... bn]
#
# Class methods:
#   solve system of equations using specified method: solve(method)
#   construst image from solutions: construct_image()
#==============================================================================
class LinearSamplingProblem(LinearSystem):
    
    def __init__(self, operatorName, kernel, rhs_vectors):
        super().__init__(asConvolutionalOperator(kernel), rhs_vectors)
        self.operatorName = operatorName
        self.kernel = kernel
        
        
    def solve(self, method, nproc=1, alpha=0.0, atol=1.0e-8, btol=1.0e-8, k=None):
        '''
        method : specified direct or iterative method for solving Ax = b
        alpha : regularization parameter
        atol : error tolerance for the linear operator
        btol : error tolerance for the right-hand side vectors
        k : number of singular values/vectors
        '''
        #======================================================================
        if method == 'lsmr':
            print('Localizing targets...')
            return super().solve_lsmr(alpha, atol, btol, nproc)
        
        elif method == 'lsqr':
            print('Localizing targets...')
            return super().solve_lsqr(alpha, atol, btol, nproc)
        
        elif method == 'svd':
            # Load or recompute the SVD of A as needed
        
            if self.operatorName == 'nfo':
                filename = 'NFO_SVD.npz'
            elif self.operatorName == 'lso':
                filename = 'LSO_SVD.npz'
        
            try:
                U, s, Vh = load_svd(filename)
                if svd_needs_recomputing(self.kernel, k, U, s, Vh):
                    U, s, Vh = compute_svd(self.kernel, k, self.operatorName)
            except IOError as err:
                print(err.strerror)
                if k is None:
                    k = input('Specify the number of singular values and vectors to compute: ')
                U, s, Vh = compute_svd(self.kernel, k, self.operatorName)
            
            print('Localizing targets...')
            return super().solve_svd(U, s, Vh, alpha, nproc)
            
    
    def construct_image(self, solutions):
        print('Constructing the image...')
        # Get machine precision
        eps = np.finfo(float).eps     # about 2e-16 (used in division
                                      # so we never divide by zero)
        if self.operatorName == 'nfo':
            Image = 1.0 / (norm(solutions, axis=0) + eps)
            
            # Normalize Image to take on values between 0 and 1
            Imin = np.min(Image)
            Imax = np.max(Image)
            Image = (Image - Imin) / (Imax - Imin + eps)
            
        elif self.operatorName == 'lso':
            Nm, Nsp = self.kernel.shape[1], self.kernel.shape[2]
            K = solutions.shape[1]
            solutions = solutions.reshape((Nsp, Nm, K))
            
            # Initialize the Image
            Image = np.zeros(Nsp)
            for i in range(K):
                indicator = norm(solutions[:, :, i], axis=1)
                Imin = np.min(indicator)
                Imax = np.max(indicator)
                indicator = (indicator - Imin) / (Imax - Imin + eps)
                Image += indicator**2
        
            # Image is defined as the root-mean-square indicator
            Image = np.sqrt(Image / K)
            
        return Image