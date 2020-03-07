from cvxopt import matrix, spmatrix, solvers
import numpy as np

class l1tf:    
    def __init__(y,lambdaaa):
        """
        Finds the solution of the l1 trend estimation problem
            minimize    (1/2)*v'*D*D'*v-y'*D'*v
            subject to  ||v||_inf <= lambda
        with variable v.
    
        Parameters
        ----------
        y : numpy.ndarray
            1-D array of original signal containing data with 'float' type.
        lambdaaa : float
            Positive regularization parameter.
    
        Returns
        -------
        cvxopt.base.matrix
            Primal optimal point.
    
        Author: Gabriel Daely
            https://github.com/daeIy
    
        This code is rewritten in Python 3.7 (CVXOPT and NumPy)
        based on l1 trend filtering algorithm by
        Kwangmoo Koh, Seung-Jean Kim and Stephen Boyd.
        https://web.stanford.edu/~boyd/papers/l1_trend_filter.html
    
        """
    
        n = len(y)
        m = n-2
    
        # Convert array y to cvxopt.spmatrix
        y = spmatrix(y,range(n),[0]*n,tc='d')
        # Create second order difference matrix
        D = spmatrix([1,-2,1]*m,
                     [j for i in range(m) for j in [i]*3],
                     [j for i in range(m) for j in [i,i+1,i+2]],tc='d')
    
        # Create P and q
        P = D * D.T
        q = D * y * (-1)
        q = matrix(q)
        # Create G and h
        G = spmatrix([1]*m+[-1]*m, range(2*m), 2*[*range(m)])
        h = matrix(lambdaaa, (2*m, 1))
        # Solve the QP problem
        res = solvers.qp(P, q, G, h)
        sol = y - D.T * res['x']
        return sol
