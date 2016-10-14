import linreggrad
import numpy as np
import pandas as pd
import random


# Regularization parameter
alpha = 0.
# Learning rate
step_size = 10000.
# Columns (features) to delete
cols = [4,11,12,13,14,19,23,33,35,36,37,42,45,46,47,53,54,55,56,60,63,79,91,93,
        96,99,112,121,123,126,127,128,129,130,131,132,133,134,135,136,137,138,
        139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,
        157,158,159,160,161]


dtrain = pd.read_csv('my_train.csv', header=None).values
dtrain_xs = dtrain[:,:162]
dtrain_xs = np.delete(dtrain_xs, cols, axis=1)
dtrain_ls = dtrain[:,162:].reshape(5652,)
dtest_xs = pd.read_csv('my_test.csv', header=None).values
dtest_xs = np.delete(dtest_xs, cols, axis=1)


lrg = linreggrad.LinRegGrad(alpha, dtrain_ls, step_size, dtrain_xs)
lrg.fit()


dtest_ys = lrg.predict(dtest_xs)
ids = np.array(['id_{}'.format(i) for i in xrange(240)])
np.savetxt('kaggle_best.csv', np.column_stack((ids, dtest_ys)),
        header='id,value', comments='', fmt='%s', delimiter=',')

