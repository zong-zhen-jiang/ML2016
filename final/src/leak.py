import csv
import os
import sys


#
DATA_DIR = '../data'
#
OUT_CSV = '../data/_leak.csv'

leak = {}
for c,row in enumerate(
        csv.DictReader(open(DATA_DIR + '/promoted_content.csv'))):
    if row['document_id'] != '':
        leak[row['document_id']] = 1

#
filename = DATA_DIR + '/page_views.csv'
for c, row in enumerate(csv.DictReader(open(filename))):
    if c % 9876 == 0:
        sys.stdout.write(
                '\rExtracting leak... (%d/%d)' % (c, 2034275449))
        sys.stdout.flush()

    if row['document_id'] not in leak:
        continue

    if leak[row['document_id']] == 1:
        leak[row['document_id']] = set()

    lu = len(leak[row['document_id']])
    leak[row['document_id']].add(row['uuid'])

sys.stdout.write('\n')

#
fo = open(OUT_CSV, 'w')
fo.write('document_id,uuid\n')
cnt = 0
for i in leak:
    if cnt % 87 == 0:
        sys.stdout.write('\rWriting leak file... (%d/%d)' % (cnt, 21938))
        sys.stdout.flush()

    if leak[i] != 1:
        cnt += 1
        tmp = list(leak[i])
        fo.write('%s,%s\n' % (i, ' '.join(tmp)))
        del tmp
fo.close()
sys.stdout.write('\n')
