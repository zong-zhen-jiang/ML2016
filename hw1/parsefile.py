import numpy as np
import pandas as pd


print 'Parsing input files...'

dtrain = pd.read_csv('data/train.csv').values[0:4320,3:27]
for month in xrange(12):
    poop = dtrain[month*360:month*360+360,0:24]
    for r in xrange(360):
        for c in xrange(24):
            if poop[r][c] == 'NR':
                poop[r][c] = '0'
    shit = np.zeros(shape=(18,480), dtype=float)
    for day in xrange(20):
        shit[0:18,day*24:day*24+24] = poop[day*18:day*18+18,0:24]
    np.savetxt('tmp/m{:02d}.csv'.format(month), shit, delimiter=',', fmt='%s')


my_train = np.zeros(shape=(471*12,9*18+1), dtype=float)
for month in xrange(12):
    shit = pd.read_csv('tmp/m{:02d}.csv'.format(month), header=None).values
    for hr in xrange(471):
        my_train[month*471+hr,0:9*18] = shit[0:18,hr:hr+9].reshape(1, 9 * 18)
        my_train[month*471+hr,9*18] = shit[9,hr+9]
np.savetxt('my_train.csv', my_train, delimiter=',', fmt='%s')


dtest = pd.read_csv('data/test_X.csv', header=None).values[0:240*18,2:11]
for r in xrange(dtest.shape[0]):
    for c in xrange(dtest.shape[1]):
        if dtest[r][c] == 'NR':
            dtest[r][c] = '0'

my_test = np.zeros(shape=(240,9*18), dtype=float)
for i in xrange(240):
    my_test[i,0:] = dtest[i*18:i*18+18,0:].reshape(1, 9 * 18)
np.savetxt('my_test.csv', my_test, delimiter=',', fmt='%s')

print 'Done parsing files!'

