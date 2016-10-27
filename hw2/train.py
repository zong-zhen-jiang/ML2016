import cPickle
import numpy as np
import pandas as pd
import poop
import sys
import time


in_file = sys.argv[1]
model_file = sys.argv[2]
print '(in_file, model_file) =\n\t(\'%s\', \'%s\')' % (in_file, model_file)


dtrain = pd.read_csv(in_file, header=None).values[:,1:]
dtrain_xs = dtrain[:,:-1]
dtrain_ys = dtrain[:,-1].reshape(len(dtrain),)
dtrain_xs = np.array(
        [np.append(d, [np.log(d[54]), np.log(d[55])]) for d in dtrain_xs])


start = time.time()
model = poop.LogisticRegression(lrate=20.0, num_iters=500000)
#model = poop.RandomForest(num_features=30, num_trees=50)
model.fit(dtrain_xs, dtrain_ys)
secs = time.time() - start
m, s = divmod(secs, 60)
h, m = divmod(m, 60)
print 'Elapsed time %d:%02d:%02d' % (h, m, s)
print 'Accuracy (training set): %f' % model.score(dtrain_xs, dtrain_ys)


with open(model_file, 'wb') as output:
    cPickle.dump(model, output, cPickle.HIGHEST_PROTOCOL)
