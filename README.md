# k*-Nearest Neighbors
Simple implementation of the k*-NN proposed by Anava and Levy in
[k*-Nearest Neighbors: From Global to Local](https://arxiv.org/pdf/1701.07266.pdf).

## Motivation
Recently, Coscrato et al. in
[NLS: an accurate and yet easy-to-interpret regression method](https://arxiv.org/pdf/1910.05206.pdf)
trained a neural network to spit out the coefficients of a local linear regression.
They sought "a method that
that is complex enough to give good predictions,
and yet gives solutions that are easy to be interpreted
without the need of using a separate interpreter".

I was a bit disappointed that they did not include the 
k*-NN method in their benchmark.
I guess, the reason therefore is that there is no implementation (that I know of).
 
k*-NN is a simple approach to locally weighted regression/classification, but we only consider regression.
The optimal weights, and the optimal number of neighbors, are efficiently found
for each data point whose value we wish to estimate.
In this respect, k*-NN is data-adaptive.

k*-NN was shown to beat the standard k-NN and the Nadarays-Watson estimator
on a collection of datasets. But it is not widespreadly used. Why?
I believe the issue is that its performance degrades drastically in presence of irrelevant covariates.
So, a preprocessing step is needed to select relevant features.
In this respect, Random Forest works very well out-of-the-box and
is robust to inclusion of irrelevant features.

## Dependencies
You will need python 3 with
[numpy](http://www.numpy.org/)
and [sklearn](http://scikit-learn.org/stable/).
If you intend to run the tests, then
[matplotlib](https://matplotlib.org/)
must be installed.

## Code
It is light-weight: A single class `kstarnn` defined in `kstarnn.py`.
It is sklearn-compatible with `fit` and `predict` methods.
The `fit` method just builds a KDTree.

The number of neighbors needed for prediction at a given point
is data-driven. Ideally, we should add one neighbor at a time.
However, no KDTree offers this functionality: We can only 
query a fix number of nearest neighbors. Stated otherwise,
there is no method `query_next`. 
So, we just query a bunch of neighbors (30 by default)
with the hope that it is sufficient. If not, a warning is issued.
This is the main weakness of the present implementation.

## Code example
Suppose `train_X` (a numpy array) and `train_Y` (a numpy vector) are the training data sets:
Inputs and labels (continuous for regression), respectively.
Then the following will fit a k*-NN model:

```python
knn = kstarnn(alpha=10.)
knn.fit(Xtrain, ytrain)
preds, ngbs = knn.predict(Xtest, return_ngbs = True)
```

With the option `return_ngbs = True` (`False` by default),
the indices of the neighbors per test data point are returned in a list.

The parameter `alpha` is usually set by cross-validation. See `test.py` for an example.

## Parameters
* alpha [default=1]
  - This is L/C where L designates the Lipschitz constant of the unknown function to regress
and C is a constant coming from the Hoeffding's inequality, see paper.
* max_num_neighbors [default=30]
  - Maximum number of neighbors to request in a query.
* copy_X_train [default=True]
  - If `True`, the training dataset is copied.

## Tests
Run `test.py` to produce the following:

![Demo](https://github.com/ldv1/kstar-NN/blob/master/test.png)

Zero-mean Gaussian noise with stand deviation of 0.1 was added to the targets.

Now we consider the same underlying sine function as a function of x_1, and add an irrelevant feature x_2.
We plot the projection onto the covariate x_1:

![Demo](https://github.com/ldv1/kstar-NN/blob/master/test_2D.png)

Random Forest does a very good job whereas k*-NN performs poorly. 

## Authors
Laurent de Vito

## License
All third-party libraries are subject to their own license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
