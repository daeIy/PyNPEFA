from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
import numpy as np

class l1tf_lm:
    def __init__(y):
        """
        Returns an upperbound of lambda. With a regularization parameter value over lambda_max, l1tf returns the best affine fit for y.
    
        Parameters
        ----------
        y : numpy.ndarray or pandas.Series
            1-D array of original signal containing data with 'float' type.
    
        Returns
        -------
        float
            Maximum value of lambda.
    
        Author: Gabriel Daely
            https://github.com/daeIy
    
        This code is rewritten in Python 3.7 (SciPy and NumPy)
        based on l1 trend filtering algorithm by
        Kwangmoo Koh, Seung-Jean Kim and Stephen Boyd.
        https://web.stanford.edu/~boyd/papers/l1_trend_filter.html
    
        """
        n = len(y)
        m = n-2
        # Convert array y to cvxopt.spmatrix
        y = csr_matrix((y, (np.array([*range(n)]),
                            np.array([0]*n))))
        # Create second order difference matrix
        D = diags([1,-2,1], [0,1,2], shape=(m,n))
        DDt = D * D.T
        Dy = D * y
    
        return norm(spsolve(DDt, Dy), np.inf)
