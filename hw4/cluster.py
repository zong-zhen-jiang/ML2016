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
import re
import sys
import word2vec


# Input arguments
input_dir = sys.argv[1]
out_csv = sys.argv[2]

#############################################################################
# Stopword removal
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
rm_stopword('docs.txt', 'docs2.txt')
sys.stdout.write('\r[Step 1/6] Removing stop words... Done :)\n')


#############################################################################
# Word to phrase
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
# Word vectors and similarities
#############################################################################

class MySentences(object):

    def __init__(self, filename):
        self.files = filename

    def __iter__(self):
        for f in self.files:
            for idx, line in enumerate(open(f)):
                yield line.split()

sentences = MySentences(['title3.txt', 'docs3.txt'])

model = Word2Vec(alpha=0.025, min_alpha=0.025, workers=8, size=300,
        min_count=2, window=20, sample=0)
model.build_vocab(sentences)
for i in range(10):
    sys.stdout.write(
            '\r[Step 3/6] Training word vectors... (%2d/10)' % (i + 1))
    sys.stdout.flush()
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha / 100
sys.stdout.write('\n')


#############################################################################
# Similar words augmentation
#############################################################################

tmp = open('title4.txt', 'w')
hist = {}
for idx, line in enumerate(open('title3.txt')):
    sys.stdout.write('\r[Step 4/6] Query expansion... (%5d/20000)' % (idx + 1))
    sys.stdout.flush()

    words_to_rm = []
    while len(line.split()) >= 5:
        try:
            unmatch = model.doesnt_match(line.split())
            line = line.replace(unmatch, '', 1)
        except ValueError:
            break
    words = []
    for w in line.split():
        if not model.vocab.has_key(w):
            continue

        c = model.vocab[w].count
        if c > 100:
            words.append(w)

            sim_w = None
            if hist.has_key(w):
                sim_w = hist[w]
            else:
                sim_w = model.most_similar(w)
                hist.update({w: sim_w})
            for i in xrange(6):
                if sim_w[i][1] > 0.4:
                    words.append(sim_w[i][0])

    tmp.write(' '.join(words) + '\n')
sys.stdout.write('\n')
tmp.close()


#############################################################################
# Prediction
#############################################################################

# (5000000, 3)
chk_idx_csv = pd.read_csv(input_dir + '/check_index.csv').values

# 
lines_of_words = []
for idx, line in enumerate(open('title4.txt')):
    lines_of_words.append(line.split())
lines_of_words = np.array(lines_of_words)


result = []
idx = 0
for r in chk_idx_csv:
    if (idx + 1) % 10 == 0:
        sys.stdout.write(
                '\r[Step 5/6] Predicting... (%7d/5000000)' % (idx + 1))
        sys.stdout.flush()
    idx += 1

    words_1 = set(lines_of_words[r[1]])
    words_2 = set(lines_of_words[r[2]])

    if len(words_1.intersection(words_2)) >= 7:
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
