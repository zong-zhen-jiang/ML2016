from csv import DictReader
from math import exp, log, sqrt

import cPickle
import csv
import operator
import sys
import time


###############################################################################
#
###############################################################################

# Path to folder containing all the original files from Kaggle
data_dir = '../data'

# Model parameters
alpha = .0130
beta = 0.0
L1 = 0.0
L2 = 0.0
D = 2 ** 26
interaction = False
epoch = 100


##############################################################################
#
##############################################################################

class ShittyFtrl(object):

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path))):
        # process id
        disp_id = row['display_id']
        ad_id = row['ad_id']

        # process clicks
        y = 0.
        if 'clicked' in row:
            if row['clicked'] == '1':
                y = 1.
            del row['clicked']

        x = []
        for key in row:
            x.append(abs(hash(key + '_' + row[key])) % D)

        #
        row = prcont_dict.get(ad_id, [])
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind == 0:
                ad_doc_id = val
            x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

        #
        row = doc_meta_dict[ad_doc_id]
        x.append(abs(hash('src_' + row[0])) % D)
        x.append(abs(hash('puber_' + row[1])) % D)

        #
        cat_id = doc_cat_dict.get(ad_doc_id, '')
        if not cat_id == '':
            x.append(abs(hash('cat_' + cat_id)) % D)
        #
        top_id = doc_top_dict.get(ad_doc_id, '')
        if not top_id == '':
            x.append(abs(hash('top_' + top_id)) % D)

        #
        row = event_dict.get(disp_id, [])
        uuid_val = -1
        for ind, val in enumerate(row):
            if ind == 0:
                uuid_val = val
            x.append(abs(hash(event_header[ind] + '_' + val)) % D)

        if (ad_doc_id in leak_uuid_dict) and\
                (uuid_val in leak_uuid_dict[ad_doc_id]):
            x.append(abs(hash('leakage_row_found_1'))%D)
        else:
            x.append(abs(hash('leakage_row_not_found'))%D)

        yield t, disp_id, ad_id, x, y


##############################################################################
#
##############################################################################

# initialize ourselves a learner
learner = ShittyFtrl(alpha, beta, L1, L2, D, interaction)


with open(data_dir + '/documents_meta.csv') as infile:
    doc_meta_csv = csv.reader(infile)
    next(doc_meta_csv)
    doc_meta_dict = {}
    prev_doc_id = ''
    for idx, row in enumerate(doc_meta_csv):
        if idx % 9876 == 0:
            sys.stdout.write('\rdocuments_meta (%7d/%d)' % (idx, 2999335))
            sys.stdout.flush()

        # 0: doc_id, 1: src_id, 2: publisher_id, 3: pub_time
        if row[0] != prev_doc_id:
            doc_meta_dict[row[0]] = [row[1], row[2]]
            prev_doc_id = row[0]

sys.stdout.write('\n')
del doc_meta_csv


with open(data_dir + '/documents_categories.csv') as infile:
    doc_cat_csv = csv.reader(infile)
    next(doc_cat_csv)
    doc_cat_dict = {}
    prev_doc_id = ''
    for idx, row in enumerate(doc_cat_csv):
        if idx % 9876 == 0:
            sys.stdout.write('\rdocuments_categories (%7d/%d)' % (idx, 5481476))
            sys.stdout.flush()

        # 0: doc_id, 1: cat_id, 2: conf
        if row[0] != prev_doc_id:
            doc_cat_dict[row[0]] = row[1]
            prev_doc_id = row[0]

sys.stdout.write('\n')
del doc_cat_csv


with open(data_dir + '/documents_topics.csv') as infile:
    doc_top_csv = csv.reader(infile)
    next(doc_top_csv)
    doc_top_dict = {}
    prev_doc_id = ''
    for idx, row in enumerate(doc_top_csv):
        if idx % 9876 == 0:
            sys.stdout.write('\rdocuments_topics (%8d/%d)' % (idx, 11325961))
            sys.stdout.flush()

        # 0: doc_id, 1: top_id, 2: conf
        if row[0] != prev_doc_id:
            doc_top_dict[row[0]] = row[1]
            prev_doc_id = row[0]

sys.stdout.write('\n')
del doc_top_csv


with open(data_dir + '/promoted_content.csv') as infile:
    prcont = csv.reader(infile)
    prcont_header = next(prcont)[1:]
    prcont_dict = {}
    for ind,row in enumerate(prcont):
        prcont_dict[row[0]] = row[1:]
        if ind % 9876 == 0:
            sys.stdout.write('\rpromoted_content (%6d/%d)' % (ind, 559584))
            sys.stdout.flush()
    sys.stdout.write('\n')
del prcont


with open(data_dir + '/events.csv') as infile:
    events = csv.reader(infile)
    next(events)
    event_header = [
            'uuid',
            'document_id',
            'platform',
            'geo_location',
            'loc_country',
            'loc_state',
            'loc_dma']
    event_dict = {}
    for ind,row in enumerate(events):
        # 0: disp_id, 1: uuid, 2: doc_id, 3: time, 4: plat, 5: geo
        tlist = [row[1], row[2], row[4]]
        loc = row[5].split('>')
        if len(loc) == 3:
            tlist.extend(loc[:])
        elif len(loc) == 2:
            tlist.extend( loc[:]+[''])
        elif len(loc) == 1:
            tlist.extend( loc[:]+['',''])
        else:
            tlist.append(['','',''])
            print 'Should not get there!'
        event_dict[row[0]] = tlist[:]
        if ind % 9876 == 0:
            sys.stdout.write('\revents (%8d/%d)' % (ind, 23120127))
            sys.stdout.flush()
    sys.stdout.write('\n')
del events


csv.field_size_limit(sys.maxsize)
with open(data_dir + '/_leak.csv') as infile:
    doc = csv.reader(infile)
    doc.next()
    leak_uuid_dict = {}
    for ind, row in enumerate(doc):
        doc_id = row[0]
        leak_uuid_dict[doc_id] = set(row[1].split(' '))
        if ind % 87 == 0:
            sys.stdout.write('\rleak (%5d/%d)' % (ind, 21938))
            sys.stdout.flush()
    sys.stdout.write('\n')
del doc


# start training
for e in range(epoch):
    t_start = time.time()
    for t, disp_id, ad_id, x, y in data(data_dir + '/_clicks_train_sp.csv', D):
        if t % 9876 == 0:
            sys.stdout.write('\r[Epoch %d/%d] %d/%d, %.1f secs' % (
                    e, epoch, t, 67237323, time.time() - t_start))
            sys.stdout.flush()

        # Predicted probability
        p = learner.predict(x)
        learner.update(x, p, y)

    sys.stdout.write('\r[Epoch %d/%d] %d/%d, %.1f secs\n' % (
            e, epoch, 67237323, 67237323, time.time() - t_start))


##############################################################################
#
##############################################################################

def predict(in_file, out_file, bitch=-1):
    out_file_ = open(data_dir + '/' + out_file, 'w')
    out_file_.write('display_id,ad_id\n')

    # {'149930': 0.128709, '153623': 0.233386, ...}
    ad_click_dict = {}
    prev_display_id = ''
    for t, disp_id, ad_id, x, y in data(data_dir + '/' + in_file, D):
        if t % 9876 == 0:
            sys.stdout.write(
                    '\rPredicting "%s"... (%d/%d)' % (out_file, t, bitch))
            sys.stdout.flush()

        p = learner.predict(x)
        if len(ad_click_dict) == 0:
            # Should only run once.
            ad_click_dict[ad_id] = p
            prev_display_id = disp_id
            continue

        if prev_display_id != disp_id:
            # Sort ad_click_dict's keys by values.
            # [('1234', 0.5566), ('9487', 0.2882), ('5278', 0.087), ...]
            fuck = sorted(
                    ad_click_dict.items(),
                    key=operator.itemgetter(1),
                    reverse=True)
            # ['1234', '9487', '5278', ...]
            fuck = [f[0] for f in fuck]
            # <display_id>, <ad_id_1> <ad_id_2> <ad_id_3> ...
            out_file_.write('%s,%s\n' % (prev_display_id, ' '.join(fuck)))

            ad_click_dict = {}

        ad_click_dict[ad_id] = p
        prev_display_id = disp_id

    fuck = sorted(
            ad_click_dict.items(),
            key=operator.itemgetter(1),
            reverse=True)
    fuck = [f[0] for f in fuck]
    out_file_.write('%s,%s\n' % (prev_display_id, ' '.join(fuck)))
    out_file_.close()

    sys.stdout.write('\n')


predict('_clicks_valid_sp.csv', '_va_out.csv', 19904410)
predict('clicks_test.csv', '_te_out.csv', 32225163)
