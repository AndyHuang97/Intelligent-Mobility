import numpy as np
import time

class HEOM():
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
        self.categorical_time = 0
        self.numerical_time = 0
        self.missing_time = 0  
        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
        self.num_ix = np.setdiff1d(self.col_ix, self.cat_ix)
        # Get the normalization scheme for numerical variables
        if normalised == "std":
            self.range = 4* np.nanstd(X, axis = 0)
        else:
            self.range = np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0)
    
    def heom(self, x, y):
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
        results_array = np.zeros(x.shape)
        
        #start = time.time() ###
        # Calculate the distance for categorical elements
        results_array[self.cat_ix] = np.not_equal(x[self.cat_ix], y[self.cat_ix]) * 1 # use "* 1" to convert it into int
        #end = time.time() ###
        #self.categorical_time += (end-start)
        # print(self.cat_ix, results_array[self.cat_ix])

        
        #start = time.time() ###
        # Calculate the distance for numerical elements
        results_array[self.num_ix] = np.abs(x[self.num_ix] - y[self.num_ix]) / self.range[self.num_ix]
        #end = time.time() ###
        #self.numerical_time += (end-start)
        # print(self.num_ix, results_array[self.num_ix])

        # Return the final result
        # Square root is not computed in practice
        # As it doesn't change similarity between instances
        return np.sum(np.square(results_array))