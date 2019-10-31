import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from kstarnn import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
label_fs = 13

np.random.seed(1234)

# data
def test_func(x, f):
    return np.cos(2*np.pi*f*x)

Ntrain = 50
f = 2
s = 1e-1 # std of noise
Xtrain = np.random.rand(Ntrain)
ytrain = test_func(Xtrain, f) + s*np.random.randn( Ntrain )
Ntest = 200
Xtest  = np.linspace(0,1,Ntest)
ytest  = test_func(Xtest, f)

# using CV
parameters = {'alpha': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600, 1000 ] }
knn = kstarnn()
cv_sets = KFold(n_splits=min(20,Ntrain), shuffle=True, random_state=123)
reg = GridSearchCV(knn, parameters, scoring='neg_mean_squared_error', cv=cv_sets, refit=False, n_jobs=-1)
reg.fit(Xtrain.reshape((-1,1)), ytrain)

# build best regressor
knn = kstarnn(**reg.best_params_)
knn.fit(Xtrain.reshape((-1,1)), ytrain)
preds, ngbs = knn.predict(Xtest.reshape(-1,1), return_ngbs = True)
num_ngbs = [ len(idx) for idx in ngbs ]

# print CV results
print("Best parameters set found on training set:")
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

# plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(3,1,1)
ax1.plot(Xtest, ytest, linewidth = 2, color = "black", label = "Ground truth")
ax1.scatter(Xtrain, ytrain, s = 30, color = "black", label = "Samples")
ax1.plot(Xtest, preds, linewidth = 2, color = "red", label = "Predictions")
ax1.set_xlabel("Covariate", fontsize=label_fs)
ax1.set_ylabel("Function values", fontsize=label_fs)
ax1.legend(prop={'size': 12})
ax2 = fig.add_subplot(3,1,2)
ax2.plot(Xtest, num_ngbs, linewidth = 2, color = "black", label = "Samples")
ax2.set_xlabel("Covariate", fontsize=label_fs)
ax2.set_ylabel("Number of neighbors", fontsize=label_fs)
ax3 = fig.add_subplot(3,1,3)
ax3.semilogx(parameters["alpha"], means, linewidth = 2, color = "black", label = "CV scores: Mean")
ax3.scatter(parameters["alpha"], means, s = 30, color = "black")
ax3.fill_between(parameters["alpha"],
                 means-stds,
                 means+stds,
                 color = "black", alpha=0.3, label = "CV scores: +/- stds")
ax3.axvline(x=reg.best_params_["alpha"], color = "red", linewidth = 2, label = r"Best $\alpha$" )
ax3.set_xlabel(r"Parameter $\alpha$", fontsize=label_fs)
ax3.set_ylabel("Neg. mean squared error", fontsize=label_fs)
handles, labels = ax3.get_legend_handles_labels()
order = [0,2,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], prop={'size': 12})

fig.align_ylabels([ ax1, ax2, ax3 ])
fig.set_tight_layout(True)
fig.savefig('foo.png', dpi=600)
plt.show()
