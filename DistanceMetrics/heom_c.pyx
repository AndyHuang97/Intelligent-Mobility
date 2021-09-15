import cython
import numpy as np

cdef class HEOM_C:
    cdef long[:] cat_ix
    cdef long[:] col_ix
    cdef long[:] num_ix
    cdef int num_cols
    cdef double[:] range

    def __init__(self, X, cat_ix, normalised="normal"):
        """ Heterogeneous Euclidean-Overlap Metric
        Distance metric class which initializes the parameters
        used in heom function
        
        Parameters
        ----------
        X : array-like of shape = [n_rows, n_features]
            Dataset that will be used with HEOM. Needs to be provided
            here because minimum and maximimum values from numerical
            columns have to be extracted
        
        cat_ix : array-like of shape = [cat_columns_number]
            List containing categorical feature indices
        
        cat_ix : array-like of shape = [x]
            List containing missing values indicators
        normalised: string
            normalises euclidan distance function for numerical variables
            Can be set as "std". Default is a column range
        Returns
        -------
        None
        """
        self.cat_ix = np.array(cat_ix)
        self.col_ix = np.array([i for i in range(X.shape[1])])
        self.num_ix = np.setdiff1d(self.col_ix, self.cat_ix)
        self.num_cols = X.shape[1]

        # Get the normalization scheme for numerical variables
        if normalised == "std":
            self.range = 4* np.nanstd(X, axis = 0)
        else:
            self.range = np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0)


    cpdef public double heom(self, double[:] x, double[:] y):
        """ Distance metric function which calculates the distance
        between two instances. Handles heterogeneous data and missing values.
        It can be used as a custom defined function for distance metrics
        in Scikit-Learn
        
        Parameters
        ----------
        x : array-like of shape = [n_features]
            First instance 
            
        y : array-like of shape = [n_features]
            Second instance
        Returns
        -------
        result: float
            Returns the result of the distance metrics function
        """
        # Initialise results' array
        cdef int i
        cdef double[:] results_array = np.zeros(self.num_cols)
        
        # Calculate the distance for categorical elements
        for i in self.cat_ix:
          if x[i] != y[i]:
            results_array[i] = 1.0
          else:
            results_array[i] = 0.0   

        
        # Calculate the distance for numerical elements
        for i in self.num_ix:
          results_array[i] = abs(x[i] - y[i]) / self.range[i]


        # Return the final result
        # Square root is not computed in practice
        # As it doesn't change similarity between instances
        cdef double result = 0
        for i in self.col_ix:
            result += results_array[i]**2
        return result