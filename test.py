import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from kstarnn import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
label_fs = 12

np.random.seed(1234)

# test functions
def cos_func(x):
    assert x.shape[1] == 1
    f = 2.
    return np.cos(2*np.pi*f*x)

def cos_func2(x):
    assert x.shape[1] == 2
    f = 2.
    return np.cos(2*np.pi*f*x[:,0])

def heavysine_func(x):
    return 4*np.sin(4*np.pi*x)-np.sign(x-0.3)-np.sign(0.72-x)
    
test_func = cos_func

# data
D = 1
Ntrain = 50
s = 1e-1 # std of noise
Xtrain = np.random.rand(D*Ntrain).reshape( (-1,D) )
ytrain = test_func(Xtrain).ravel() + s*np.random.randn( Ntrain )
Ntest = 1000
if D == 1:
    Xtest  = np.linspace(0,1,Ntest).reshape((-1,1))
else:
    X1test, X2test = np.meshgrid( np.linspace(0,1,Ntest), np.linspace(0,1,Ntest), indexing='ij') 
    Xtest = np.c_[X1test.ravel(), X2test.ravel()]
ytest  = test_func(Xtest).ravel()

# using CV
print("CV for k*-NN")
parameters = {'alpha': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600, 1000 ] }
cv_sets = KFold(n_splits=min(20,Ntrain), shuffle=True, random_state=123)
reg = GridSearchCV(kstarnn(), parameters, scoring='neg_mean_squared_error', cv=cv_sets, refit=False, n_jobs=-1)
reg.fit(Xtrain, ytrain)

# build best k*-NN regressor
print("k*-NN prediction")
knn = kstarnn(**reg.best_params_)
knn.fit(Xtrain, ytrain)
preds, ngbs = knn.predict(Xtest, return_ngbs = True)
num_ngbs = [ len(idx) for idx in ngbs ]
knn_rms = np.sqrt(mean_squared_error(ytest, preds))

# print K*-NN CV results
print("K*-NN: Best parameters set found on training set:")
print()
print(reg.best_params_)
print()
print("Grid scores on training set:")
print()
means = reg.cv_results_['mean_test_score']
stds = reg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, reg.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# CV for Random Forest
print("CV for Random Forest")
param_grid = { "n_estimators": np.arange(140,210,20),
               "min_samples_leaf": np.arange(2,5),
               "min_samples_split": np.arange(2,8,2),
               "max_depth": np.arange(3,18,2),
              }
reg = GridSearchCV(RandomForestRegressor(random_state=1), \
                   param_grid, \
                   scoring='neg_mean_squared_error', \
                   cv=cv_sets, \
                   n_jobs=-1)
reg.fit(Xtrain, ytrain)
print(reg.best_params_)
rf_preds = reg.predict(Xtest)
rf_rms = np.sqrt(mean_squared_error(ytest, rf_preds))

#results
print("RMS:")
print("K*-NN: {:.1e}, RF: {:.1e}".format(knn_rms, rf_rms))

# plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(3,1,1)
ax1.plot   (Xtest [:,0], ytest [:], linewidth = 3, color = "black", label = "Ground truth")
ax1.scatter(Xtrain[:,0], ytrain[:], s = 30, color = "black", label = "Samples")
ax1.plot(Xtest[:,0], preds, linewidth = 3, color = "red", alpha=1., label = "k*-NN predictions")
ax1.plot(Xtest[:,0], rf_preds, linewidth = 3, color = "green", alpha=1., label = "Random Forest predictions")
ax1.set_xlabel(r"Covariate $x_1$", fontsize=label_fs)
ax1.set_ylabel("Function values", fontsize=label_fs)
ax1.legend(prop={'size': 12})
ax2 = fig.add_subplot(3,1,2, sharex=ax1)
ax2.plot(Xtest[:,0], num_ngbs, linewidth = 2, color = "black", label = "Samples")
ax2.set_xlabel("Covariate", fontsize=label_fs)
ax2.set_ylabel("Number of neighbors", fontsize=label_fs)
ax3 = fig.add_subplot(3,1,3)
ax3.semilogx(parameters["alpha"], means, linewidth = 2, color = "black", label = "CV scores: Mean")
ax3.scatter(parameters["alpha"], means, s = 30, color = "black")
ax3.fill_between(parameters["alpha"],
                 means-stds,
                 means+stds,
                 color = "black", alpha=0.3, label = "CV scores: +/- stds")
ax3.axvline(x=knn.get_params()["alpha"], color = "red", linewidth = 2, label = r"Best $\alpha$" )
ax3.set_xlabel(r"Parameter $\alpha$", fontsize=label_fs)
ax3.set_ylabel("Neg. mean squared error", fontsize=label_fs)
handles, labels = ax3.get_legend_handles_labels()
order = [0,2,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], prop={'size': 12})

fig.align_ylabels([ ax1, ax2, ax3 ])
fig.set_tight_layout(True)
fig.savefig('foo.png', dpi=600)
plt.show()
