import numpy as np


class DecisionTree(object):

    def __init__(self):
        return

    def fit(self, xs, ys):
        self.tree = self.make_tree(np.c_[xs, ys])
        return

    def gini(self, num0s_l, num1s_l, num0s_r, num1s_r):
        size_l = num0s_l + num1s_l
        size_r = num0s_r + num1s_r

        p0_l = float(num0s_l) / (num0s_l + num1s_l)
        p1_l = 1 - p0_l
        gini_l = 1 - p0_l ** 2 - p1_l ** 2

        p0_r = float(num0s_r) / (num0s_r + num1s_r)
        p1_r = 1 - p0_r
        gini_r = 1 - p0_r ** 2 - p1_r ** 2

        gini = (size_l * gini_l + size_r * gini_r) / (size_l + size_r)
        # print 'L(%.2f, %.2f, %.2f), R(%.2f, %.2f, %.2f), %.3f' %\
        #         (p0_l, p1_l, gini_l, p0_r, p1_r, gini_r, gini)
        return gini

    def make_tree(self, data):
        ''' Recursively build a tree. '''
        if np.all(data[:,-1] == data[0,-1], axis=0):
            # print 'All data of the same class'
            return ('leaf', data[0][-1])

        num_feats = len(data[0]) - 1
        gini_min = 55.66
        sp_idx = -1
        sp_th = 7788.
        for i in xrange(num_feats):
            (gini, th) = self.split(data, i)
            if gini < gini_min:
                gini_min = gini
                sp_idx = i
                sp_th = th

        # print '(%d, %.2f)' % (sp_idx, sp_th)
        if sp_idx == -1:
            num0s = np.sum([1 if d[-1] == 0 else 0 for d in data])
            num1s = len(data) - num0s
            p0 = float(num0s) / (num0s + num1s)
            p1 = 1 - p0
            return ('leaf', p0, p1)
        else:
            data_l = np.array([d for d in data if d[sp_idx] <= sp_th])
            data_r = np.array([d for d in data if d[sp_idx] > sp_th])
            return ('node', sp_idx, sp_th,
                    self.make_tree(data_l),
                    self.make_tree(data_r))

    def predict(self, x):
        node = self.tree
        while True:
            if node[0] == 'leaf':
                if len(node) == 2:
                    return node[1]
                elif len(node) == 3:
                    return 0 if node[1] >= node[2] else 1
                else:
                    raise AssertionError('Fuck you!', node)
            node = node[3] if x[node[1]] <= node[2] else node[4]
        raise AssertionError('Fuck you!')

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

    def split(self, data, idx):
        tmp = data[:,[idx,-1]]
        tmp = tmp[tmp[:,0].argsort()]

        if tmp[0][0] == tmp[-1][0]:
            return (5566., 5566.)

        for i in xrange(len(tmp) - 1):
            if tmp[i][0] != tmp[i + 1][0]:
                sp = i
                break

        num0s_l = np.sum(
                [1 if tmp[i][1] == 0 else 0 for i in xrange(sp + 1)])
        num1s_l = sp + 1 - num0s_l
        num0s_r = np.sum(
                [1 if tmp[i][1] == 0 else 0 for i in xrange(sp + 1, len(tmp))])
        num1s_r = len(tmp) - sp - 1 - num0s_r

        gini_min = self.gini(num0s_l, num1s_l, num0s_r, num1s_r)
        gini_min_idx = sp

        for sp in xrange(sp + 1, len(tmp) - 1):
            if tmp[sp][1] == 0:
                num0s_l += 1
                num0s_r -= 1
            elif tmp[sp][1] == 1:
                num1s_l += 1
                num1s_r -= 1
            else:
                raise AssertionError('Fuck you!')

            if tmp[sp][0] == tmp[sp + 1][0]:
                continue

            gini = self.gini(num0s_l, num1s_l, num0s_r, num1s_r)
            if gini < gini_min:
                gini_min = gini
                gini_min_idx = sp

        th = (tmp[gini_min_idx][0] + tmp[gini_min_idx + 1][0]) * 0.5
        return (gini_min, th)
