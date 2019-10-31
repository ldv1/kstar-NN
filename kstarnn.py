import numpy as np
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator, RegressorMixin

class kstarnn(BaseEstimator, RegressorMixin):
    def __init__(self, alpha = 1., max_num_neighbors = 30, copy_X_train=True):
        super().__init__()
        
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.copy_X_train = copy_X_train
        self.max_num_neighbors = max_num_neighbors
        
    def fit(self, X, y):
        self.tree = KDTree(X, leaf_size=20)
        self.train_size = X.shape[0]
        if self.copy_X_train:
            self.Xtrain, self.ytrain = X.copy(), y.copy()
        else:
            self.Xtrain, self.ytrain = X, y
        self.Ntrain = self.Xtrain.shape[0]
        self.max_num_neighbors = min(self.Ntrain,self.max_num_neighbors)
        
    def predict(self, X, return_ngbs = False):
        print_warning = True
        Ntest = X.shape[0]
        
        preds = np.zeros(Ntest)
        if return_ngbs:
            ngbs = []
        
        for n in range(Ntest):
            dists, idx = self.tree.query(np.reshape(X[n,:],(-1,1)), k=self.max_num_neighbors)
            beta = self.alpha*dists[0]
            lbda = beta[0]+1
            k = 0
            while ( lbda > beta[k] ):
                if k == self.max_num_neighbors-1:
                    if print_warning:
                        print("too few neighbors! (using {}, alpha={})" \
                              .format(self.max_num_neighbors, self.alpha))
                        print_warning = False
                    break
                
                k += 1
                # compute lambda based on beta_1, .., beta_k
                lbda = 1./k*( np.sum(beta[:k]) + \
                              np.sqrt( k + np.sum(beta[:k])**2 - k*np.sum(beta[:k]**2) ) \
                            )
            # we have lbda <= beta[k],
            # and we use all neighbors up to k (not included)
            w = lbda - beta[:k]
            assert np.all(w > 0.)
            w = w / np.sum(w)
            preds[n] = np.dot(w,self.ytrain[ idx[0][:k] ])
            if return_ngbs:
                ngbs.append(idx[0][:k])
        
        if return_ngbs:
            return (preds, ngbs)
        else:
            return preds
