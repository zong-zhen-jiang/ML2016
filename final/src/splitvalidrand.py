from ml_metrics import mapk

import numpy as np
import pandas as pd


data_path = '/home/asdfghjkl20203/mlfinal/outbrain/data'


#
ori_train_csv = pd.read_csv(data_path + '/clicks_train.csv')

ids = ori_train_csv.display_id.unique()
valid_ids = np.random.choice(ids, size=len(ids)//5, replace=False)

train_csv = ori_train_csv[~ori_train_csv.display_id.isin(valid_ids)]
valid_csv = ori_train_csv[ori_train_csv.display_id.isin(valid_ids)]

train_csv.to_csv(data_path + '/../clicks_train.csv', index=False)
valid_csv.to_csv(data_path + '/../clicks_valid.csv', index=False)
