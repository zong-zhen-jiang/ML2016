import linreggrad
import numpy as np
import pandas as pd
import random


# Regularization parameter
alpha = 0.
# Learning rate
step_size = 10000.
# Columns (features) to delete
cols = []


dtrain = pd.read_csv('my_train.csv', header=None).values
dtrain_xs = dtrain[:,:162]
dtrain_xs = np.delete(dtrain_xs, cols, axis=1)
dtrain_ls = dtrain[:,162:].reshape(5652,)
dtest_xs = pd.read_csv('my_test.csv', header=None).values
dtest_xs = np.delete(dtest_xs, cols, axis=1)


lrg = linreggrad.LinRegGrad(alpha, dtrain_ls, step_size, dtrain_xs)
lrg.fit(max_iter=10000)


dtest_ys = lrg.predict(dtest_xs)
ids = np.array(['id_{}'.format(i) for i in xrange(240)])
np.savetxt('linear_regression.csv', np.column_stack((ids, dtest_ys)),
        header='id,value', comments='', fmt='%s', delimiter=',')

