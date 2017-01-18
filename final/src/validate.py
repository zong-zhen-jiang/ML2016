from ml_metrics import mapk

import numpy as np
import pandas as pd


# Absolute path to folder containing all the original files from Kaggle
data_dir = '../data'

#
print 'Reading predicted result...'
pred_csv = pd.read_csv(data_dir + '/_va_out.csv')
pred_ad_ids = pred_csv.ad_id.values
pred_ad_ids = [map(int, ad_ids.split()) for ad_ids in pred_ad_ids]
pred_ad_ids = np.array(pred_ad_ids)

#
print 'Reading answer...'
valid_csv = pd.read_csv(data_dir + '/_clicks_valid_sp.csv')
y = valid_csv[valid_csv.clicked==1].ad_id.values
y = y.reshape((len(y), 1))

#
print 'Scoring...'
print 'MAP@12: %f' % (mapk(y, pred_ad_ids, k=12))
