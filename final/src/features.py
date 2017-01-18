import collections
import csv
import sys


#
data_dir = '../data'


###############################################################################
#
###############################################################################

with open(data_dir + '/clicks_train.csv') as infile:
    tmp = csv.reader(infile)
    next(tmp)
    nb_ads_dict = {}
    ad_cnt = 0
    prev_display_id = ''
    for ind, row in enumerate(tmp):
        if ind % 9876 == 0:
            sys.stdout.write(
                    '\r"clicks_train.csv" (%d/%d)' % (ind, 87141732))
            sys.stdout.flush()

        if prev_display_id == '':
            # Should only run once.
            prev_display_id = row[0]
            ad_cnt = 1
            continue

        # 0: display_id, 1: ad_id, 2: clicked
        if prev_display_id != row[0]:
            nb_ads_dict[prev_display_id] = ad_cnt
            prev_display_id = row[0]
            ad_cnt = 1
        else:
            ad_cnt += 1

    nb_ads_dict[prev_display_id] = ad_cnt
    sys.stdout.write('\n')

with open(data_dir + '/clicks_test.csv') as infile:
    tmp = csv.reader(infile)
    next(tmp)
    ad_cnt = 0
    prev_display_id = ''
    for ind, row in enumerate(tmp):
        if ind % 9876 == 0:
            sys.stdout.write(
                    '\r"clicks_test.csv" (%d/%d)' % (ind, 32225163))
            sys.stdout.flush()

        if prev_display_id == '':
            # Should only run once.
            prev_display_id = row[0]
            ad_cnt = 1
            continue

        # 0: display_id, 1: ad_id, 2: clicked
        if prev_display_id != row[0]:
            if nb_ads_dict.has_key(prev_display_id):
                print 'Fuck you!'
            nb_ads_dict[prev_display_id] = ad_cnt
            prev_display_id = row[0]
            ad_cnt = 1
        else:
            ad_cnt += 1

    nb_ads_dict[prev_display_id] = ad_cnt
    sys.stdout.write('\n')
del tmp


###############################################################################
#
###############################################################################

doc_cat_dict = {}
prev_doc_id = ''
doc_cat_csv = open(data_dir + '/documents_categories.csv')
for idx, row in enumerate(csv.DictReader(doc_cat_csv)):
    if idx % 9876 == 0:
        sys.stdout.write(
                '\r"documents_categories.csv" (%d/%d)' % (idx, 5481476))
        sys.stdout.flush()

    doc_id = row['document_id']
    cat_id = row['category_id']
    # conf = row['confidence_level']
    if doc_id != prev_doc_id:
        doc_cat_dict[doc_id] = cat_id
        prev_doc_id = doc_id

sys.stdout.write('\n')
doc_cat_csv.close()


###############################################################################
#
###############################################################################

doc_meta_dict = {}
doc_meta_csv = open(data_dir + '/documents_meta.csv')
for idx, row in enumerate(csv.DictReader(doc_meta_csv)):
    if idx % 9876 == 0:
        sys.stdout.write('\r"documents_meta.csv" (%d/%d)' % (idx, 2999335))
        sys.stdout.flush()

    # 0: doc_id, 1: src_id, 2: publisher_id, 3: pub_time
    doc_meta_dict[row['document_id']] = [row['source_id'], row['publisher_id']]

sys.stdout.write('\n')
doc_meta_csv.close()


###############################################################################
#
###############################################################################

doc_top_dict = {}
prev_doc_id = ''
doc_top_csv = open(data_dir + '/documents_topics.csv')
for idx, row in enumerate(csv.DictReader(doc_top_csv)):
    if idx % 9876 == 0:
        sys.stdout.write(
                '\r"documents_topics.csv" (%d/%d)' % (idx, 11325961))
        sys.stdout.flush()

    doc_id = row['document_id']
    top_id = row['topic_id']
    # conf = row['confidence_level']
    if doc_id != prev_doc_id:
        doc_top_dict[doc_id] = top_id
        prev_doc_id = doc_id

sys.stdout.write('\n')
doc_top_csv.close()


###############################################################################
#
###############################################################################

# {ad_id: [doc_id, camp_id, adtiser_id], ...}
event_dict = {}
event_csv = open(data_dir + '/events.csv')
for idx, row in enumerate(csv.DictReader(event_csv)):
    if idx % 9876 == 0:
        sys.stdout.write('\r"events.csv" (%d/%d)' % (idx, 23120127))
        sys.stdout.flush()

    display_id = row['display_id']
    tmp = [row['uuid'],
            row['document_id'],
            row['platform'],
            row['geo_location']]
    loc = row['geo_location'].split('>')
    if len(loc) == 3:
        tmp.extend(loc)
    elif len(loc) == 2:
        tmp.extend(loc + [''])
    elif len(loc) == 1:
        tmp.extend(loc + ['', ''])
    else:
        tmp.extend(['', '', ''])
    event_dict[display_id] = tmp

sys.stdout.write('\n')
event_csv.close()


###############################################################################
#
###############################################################################

leak_dict = {}
csv.field_size_limit(sys.maxsize)
with open(data_dir + '/_leak.csv') as infile:
    leak = csv.reader(infile)
    leak.next()
    leak_dict = {}
    for idx, row in enumerate(leak):
        if idx % 123 == 0:
            sys.stdout.write('\r"_leak.csv" (%d/%d)' % (idx, 21938))
            sys.stdout.flush()
        doc_id = row[0]
        leak_dict[doc_id] = set(row[1].split(' '))
del leak
sys.stdout.write('\n')


###############################################################################
#
###############################################################################

# {ad_id: [doc_id, camp_id, adtiser_id], ...}
prcont_dict = {}
prcont_csv = open(data_dir + '/promoted_content.csv')
for idx, row in enumerate(csv.DictReader(prcont_csv)):
    if idx % 9876 == 0:
        sys.stdout.write(
                '\r"promoted_content.csv" (%d/%d)' % (idx, 559584))
        sys.stdout.flush()

    ad_id = row['ad_id']
    prcont_dict[ad_id] = [
            row['document_id'],
            row['campaign_id'],
            row['advertiser_id']]

sys.stdout.write('\n')
prcont_csv.close()


###############################################################################
#
###############################################################################

feat_dict = {}
next_feat_idx = 1
def get_feat_idx(feat_str):
    global next_feat_idx
    if not feat_dict.has_key(feat_str):
        feat_dict[feat_str] = next_feat_idx
        next_feat_idx += 1
    return feat_dict[feat_str]


def to_ffm(clicks_csv, ffm_file, bitch=999999):
    infile = open(data_dir + '/' + clicks_csv)
    outfile = open(data_dir + '/' + ffm_file, 'w')
    outfile.write(
            'clicked,' +
            'disp,' +
            'ad,' +
            'ad_doc,' +
            'camp,' +
            'ader,' +
            'uuid,' +
            'plat,' +
            'loc,' +
            'loc_c,' +
            'loc_s,' +
            'loc_d,' +
            'ad_doc_src,' +
            'ad_doc_puber,' +
            'ad_doc_cat,' +
            'ad_doc_top,' +
            'nb_ads,' +
            'leak\n')

    train = csv.reader(infile)
    train.next()
    for idx, row in enumerate(train):
        if idx % 9876 == 0:
            sys.stdout.write('\r"%s" (%8d/%d), # features: %d' %
                    (ffm_file, idx, bitch, len(feat_dict)))
            sys.stdout.flush()

        #
        display_id = row[0]
        ad_id = row[1]
        clicked = int(row[2]) if len(row) == 3 else 87
        # Promoted content
        tmp = prcont_dict[ad_id]
        ad_doc_id = tmp[0]
        camp_id = tmp[1]
        ader_id = tmp[2]
        #
        tmp = doc_meta_dict[ad_doc_id]
        ad_doc_src = tmp[0]
        ad_doc_puber = tmp[1]
        # Category and topic of ad's landing doc
        ad_doc_cat = doc_cat_dict.get(ad_doc_id, '')
        ad_doc_top = doc_top_dict.get(ad_doc_id, '')
        # Events
        tmp = event_dict[display_id]
        uuid = tmp[0]
        doc_id = tmp[1]
        platform = tmp[2]
        geo_loc = tmp[3]
        loc_country = tmp[4]
        loc_state = tmp[5]
        loc_dma = tmp[6]
        #
        nb_ads = nb_ads_dict[display_id]
        #
        if (ad_doc_id in leak_dict) and (uuid in leak_dict[ad_doc_id]):
            in_leak = 1
        else:
            in_leak = 0

        #
        out_row = []
        out_row.append(clicked)
        #
        out_row.append(get_feat_idx('disp_' + display_id))
        out_row.append(get_feat_idx('ad_' + ad_id))
        #
        out_row.append(get_feat_idx('ad_doc_' + ad_doc_id))
        out_row.append(get_feat_idx('camp_' + camp_id))
        out_row.append(get_feat_idx('ader_' + ader_id))
        #
        out_row.append(get_feat_idx('uuid_' + uuid))
        out_row.append(get_feat_idx('plat_' + platform))
        out_row.append(get_feat_idx('loc_' + geo_loc))
        out_row.append(get_feat_idx('loc_country_' + loc_country))
        out_row.append(get_feat_idx('loc_state_' + loc_state))
        out_row.append(get_feat_idx('loc_dma_' + loc_dma))
        #
        out_row.append(get_feat_idx('ad_doc_src_' + ad_doc_src))
        out_row.append(get_feat_idx('ad_doc_puber_' + ad_doc_puber))
        #
        out_row.append(get_feat_idx('ad_doc_cat_' + ad_doc_cat))
        out_row.append(get_feat_idx('ad_doc_top_' + ad_doc_top))
        #
        out_row.append(get_feat_idx('nb_ads_' + str(nb_ads)))
        out_row.append(get_feat_idx('leak_' + str(in_leak)))

        #
        outfile.write('%s\n' % (','.join(str(x) for x in out_row)))

    sys.stdout.write('\n')
    del train

    outfile.close()
    infile.close()


###############################################################################
#
###############################################################################

to_ffm('_clicks_train_sp.csv', '_tr_sp.csv', 67237323)
to_ffm('_clicks_valid_sp.csv', '_va_sp.csv', 19904410)
to_ffm('clicks_train.csv', '_tr.csv', 87141733)
to_ffm('clicks_test.csv', '_te.csv', 32225163)
