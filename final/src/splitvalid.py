import csv
import sys


DATA_DIR = '../data'


###############################################################################
#
###############################################################################

event_dict = {}
event_csv = open(DATA_DIR + '/events.csv')
# min_ts = 5566789
# max_ts = 0
for idx, row in enumerate(csv.DictReader(event_csv)):
    if idx % 9876 == 0:
        sys.stdout.write('\rReading timestamp... (%d/%d)' % (idx, 23120127))
        sys.stdout.flush()

    display_id = row['display_id']
    ts = int(row['timestamp'])
    # min_ts = min(ts, min_ts)
    # max_ts = max(ts, max_ts)
    event_dict[display_id] = ts

sys.stdout.write('\n')
event_csv.close()


###############################################################################
#
###############################################################################

ori_train_csv = open(DATA_DIR + '/clicks_train.csv')
sp_train_csv = open(DATA_DIR + '/_clicks_train_sp.csv', 'w')
sp_train_csv.write('display_id,ad_id,clicked\n')
sp_valid_csv = open(DATA_DIR + '/_clicks_valid_sp.csv', 'w')
sp_valid_csv.write('display_id,ad_id,clicked\n')

prev_display_id = ''
to_valid = False
cnt = 0
for idx, row in enumerate(csv.DictReader(ori_train_csv)):
    if idx % 9876 == 0:
        sys.stdout.write('\rSplitting... (%d/%d)' % (idx, 87141733))
        sys.stdout.flush()

    display_id = row['display_id']
    if display_id != prev_display_id:
        prev_display_id = display_id
        ts = event_dict[display_id]
        if ts >= 11 * 24 * 60 * 60 * 1000:
            to_valid = True
        else:
            to_valid = True if cnt % 10 == 0 else False
            cnt += 1

    tmp = '%s,%s,%s\n' % (row['display_id'], row['ad_id'], row['clicked'])
    if to_valid:
        sp_valid_csv.write(tmp)
    else:
        sp_train_csv.write(tmp)

sys.stdout.write('\n')
ori_train_csv.close()
sp_train_csv.close()
sp_valid_csv.close()
