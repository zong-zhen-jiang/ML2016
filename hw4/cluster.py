# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import collections
import itertools
import numpy as np
import pandas as pd
import random
import re
import sys
import word2vec


# Input arguments
input_dir = sys.argv[1]
out_csv = sys.argv[2]


#############################################################################
# 1. Stopword removal
#############################################################################

def rm_stopword(in_file_path, out_file_path):
    in_file = open(in_file_path, 'r')
    out_file = open(out_file_path, 'w')
    deli = '\_|\.|\||\´|\”|\’|\!|\&|\\\|\$|\-|\<|\>|\?|\,|\*|\n|\ |\(|\)|' +\
            '\:|\"|\+|\=|\'|\/|\`|\[|\]|\{|\}|\^|\;|\~'
    stwds = set([line.strip() for idx, line
            in enumerate(open('stopwords1.txt', 'r'))])
    stwds_2 = [line.strip() for idx, line
            in enumerate(open('stopwords2.txt', 'r'))]
    stwds.update(stwds_2)
    for idx, line in enumerate(in_file):
        words = [w for w in re.split(deli, line.lower())
                if (w not in stwds) and (w != '') and (not w.isdigit())]
        out_file.write(' '.join(words) + '\n')
    in_file.close()
    out_file.close()

sys.stdout.write('[Step 1/6] Removing stop words...')
sys.stdout.flush()
rm_stopword(input_dir + '/title_StackOverflow.txt', 'title2.txt')
rm_stopword(input_dir + '/docs.txt', 'docs2.txt')
sys.stdout.write('\r[Step 1/6] Removing stop words... Done :)\n')


#############################################################################
# 2. Word to phrase
#############################################################################

sys.stdout.write('[Step 2/6] Word to phrase...')
sys.stdout.flush()

word2vec.word2phrase('title2.txt', 'title2.5.txt', verbose=False)
word2vec.word2phrase('docs2.txt', 'docs2.5.txt', verbose=False)

tmp = open('title3.txt', 'w')
for idx, line in enumerate(open('title2.5.txt')):
    tmp.write(line.translate(None, ''.join(['_'])))
tmp.close()

tmp = open('docs3.txt', 'w')
for idx, line in enumerate(open('docs2.5.txt')):
    tmp.write(line.translate(None, ''.join(['_'])))
tmp.close()

sys.stdout.write('\r[Step 2/6] Word to phrase... Done :)\n')


#############################################################################
# 3. BoW
#############################################################################

sys.stdout.write('[Step 3/6] BoW...')
sys.stdout.flush()

lines = []
for idx, line in enumerate(open('title2.txt')):
    lines.append(unicode(line, errors='ignore'))
lines = np.array(lines)

vectorizer = CountVectorizer(max_features=500)
feat = vectorizer.fit_transform(lines)
feature = feat.toarray()

sys.stdout.write('\r[Step 3/6] BoW... Done :)\n')


#############################################################################
# 4. K-means
#############################################################################

clusterers = []
nb_kmeans = 15
for fuck in xrange(nb_kmeans):
    sys.stdout.write(
            '\r[Step 4/6] Training K-means... (%2d/%2d)' % \
                    (fuck + 1, nb_kmeans))
    sys.stdout.flush()

    tmp = [random.randint(0, 19999)]
    for c in xrange(19):
        while (True):
            i = random.randint(0, 19999)
            if np.linalg.norm(feature[i]) == 0.0:
                continue
            is_ortho = True
            for j in tmp:
                ker = abs(np.dot(feature[i], feature[j]))
                if ker != 0.0:
                    is_ortho = False
                    break
            if is_ortho:
                tmp.append(i)
                break
    # print '20 ortho-centers: %s' % (str(tmp))

    init_centers = [feature[i] for i in tmp]
    init_centers.append(np.zeros(feature.shape[1]))
    init_centers = np.array(init_centers)

    # print 'K-means started, this could take minutes Q_Q'
    cluster = KMeans(init=init_centers, n_clusters=21, n_init=1, verbose=0)
    cluster.fit(feat)
    clusterers.append(cluster)
sys.stdout.write('\n')


#############################################################################
# 5. Prediction
#############################################################################

# (5000000, 3)
chk_idx_csv = pd.read_csv(input_dir + '/check_index.csv').values

# (20000,)
labels = [c.labels_ for c in clusterers]

# (5000000,)
result = []
idx = 0
for r in chk_idx_csv:
    if (idx + 1) % 10 == 0:
        sys.stdout.write(
                '\r[Step 5/6] Predicting... (%7d/5000000)' % (idx + 1))
        sys.stdout.flush()
    idx += 1

    cnt_0 = 0
    cnt_1 = 0
    for l in labels:
        l_1 = l[r[1]]
        l_2 = l[r[2]]
        if l_1 == 20 or l_2 == 20:
            cnt_0 += 1
        elif l_1 == l_2:
            cnt_1 += 1
        else:
            cnt_0 += 1
    if cnt_1 > cnt_0:
        result.append(1)
    else:
        result.append(0)
sys.stdout.write('\n')
result = np.array(result)


ids = np.array([i for i in xrange(5000000)])
sys.stdout.write('[Step 6/6] Writing CSV...')
sys.stdout.flush()
np.savetxt(out_csv, np.column_stack((ids, result)),
        header='ID,Ans', comments='', fmt='%s', delimiter=',')
sys.stdout.write('\r[Step 6/6] Writing CSV... Done :)\n')
