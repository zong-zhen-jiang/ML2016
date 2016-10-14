import numpy as np
import sys
import time


class LinRegGrad(object):

    def __init__(self, alpha, labels, step_size, xs):
        self.alpha = alpha
        self.e_prev = sys.float_info.max
        self.iter_cnt = 1
        self.ls = labels
        self.step_size = step_size
        self.w = np.random.randn(len(xs[0]) + 1)
        self.xs = np.array([np.insert(x, 0, 1) for x in xs])

    def err_mse(self, labels, xs):
        if labels is None or xs is None:
            labels = self.ls
            ys = np.array([np.dot(x, self.w) for x in self.xs])
        else:
            ys = self.predict(xs)
        return np.mean((labels - ys) ** 2)

    def err_reg(self):
        w2 = self.w.copy()
        w2[0] = 0.
        return self.alpha * np.sum(w2 ** 2)

    def fit(self, max_iter=10000000, stop_de=1e-08):
        start = time.time()
        for i in xrange(max_iter):
            self.iterate()
            if self.de < 0 and abs(self.de) < stop_de:
                break
        (m, s) = divmod(int(time.time() - start), 60)
        (h, m) = divmod(m, 60)
        print 'Spent {}:{:02d}:{:02d}'.format(h, m, s)

    def iterate(self):
        e = 0.
        eg = np.zeros(self.w.shape)
        for (x, l) in zip(self.xs, self.ls):
            y = np.dot(x, self.w)
            e += (y - l) ** 2
            eg += 2 * (y - l) * x
        e /= len(self.xs)
        eg /= len(self.xs)

        w2 = self.w.copy()
        w2[0] = 0.
        e += self.alpha * np.sum(w2 ** 2)
        eg += 2 * self.alpha * w2

#        print 'Estimated dE: {}'.format(-self.step_size * np.linalg.norm(eg))
        self.e = e
        self.de = e - self.e_prev
        print 'Iter #{}, E: {:.4f}, dE: {:.18f}, step: {:.10f}'.format(
                self.iter_cnt, self.e, self.de, self.step_size)
        self.iter_cnt += 1

        if e > self.e_prev:
            print 'Oops!'
            self.e_prev += 1
            self.step_size *= .5
            self.w = self.w_prev.copy()
            return

        self.e_prev = e
        self.step_size *= 1.01
        self.w_prev = self.w.copy()
        self.w -= self.step_size * eg / np.linalg.norm(eg)

    def predict(self, xs):
        xs = np.array([np.insert(x, 0, 1) for x in xs])
        return np.array([np.dot(x, self.w) for x in xs])

