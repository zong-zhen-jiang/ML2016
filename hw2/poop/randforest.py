from dectree import DecisionTree
import numpy as np
import threading


class RandomForest(object):

    def __init__(self, num_features, num_trees):
        self.num_features = num_features
        self.num_trees = num_trees
        return

    def fit(self, xs, ys):
        self.trees = []
        for i in xrange(self.num_trees):
            # Randomly sample N data, with replacement.
            rows = np.random.choice(len(xs), size=len(xs), replace=True)
            rows = np.unique(rows)
            sample_xs = xs[rows]
            sample_ys = ys[rows]

            # Randomly select m features (m <= M).
            cols = np.random.choice(
                    len(xs[0]), size=self.num_features, replace=False)
            sample_xs = sample_xs[:,cols]

            # Build a tree.
            print 'Building tree (%d / %d)...' % (i + 1, self.num_trees)
            tree = DecisionTree()
            tree.fit(sample_xs, sample_ys)
            self.trees.append((cols, tree))
        return

    def predict(self, x):
        num0s = 0
        num1s = 0
        for (cols, tree) in self.trees:
            a = tree.predict(x[cols])
            if a == 0.:
                num0s += 1
            elif a == 1.:
                num1s += 1
            else:
                raise AssertionError('Fuck you!', a)
        return 0 if num0s >= num1s else 1

    def predict2(self, xs):
        as_ = []
        for x in xs:
            as_.append(self.predict(x))
        return np.array(as_)

    def score(self, xs, ys):
        acc = 0.
        for (x, y) in zip(xs, ys):
            acc += 1. if self.predict(x) == y else 0.
        return acc / len(ys)
