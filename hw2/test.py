import cPickle
import numpy as np
import pandas as pd
import poop
import sys


model_file = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[3]
print '(model_file, test_file, out_file) =\n\t(\'%s\', \'%s\', \'%s\')' % \
        (model_file, test_file, out_file)


test_xs = pd.read_csv(test_file, header=None).values[:,1:]
test_xs = np.array(
        [np.append(d, [np.log(d[54]), np.log(d[55])]) for d in test_xs])


with open(model_file, 'rb') as inputt:
    model = cPickle.load(inputt)
test_as = model.predict2(test_xs)


ids = np.array(['{}'.format(i + 1) for i in xrange(600)])
np.savetxt(out_file, np.column_stack((ids, test_as)),
        header='id,label', comments='', fmt='%s', delimiter=',')
