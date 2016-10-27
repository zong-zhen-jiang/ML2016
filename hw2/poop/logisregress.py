import numpy as np
import sys
import time


class LogisticRegression(object):

    def __init__(self, alpha=0.0, lrate=10.1, num_iters=500000):
        self.alpha = alpha
        self.init_lrate = lrate
        self.num_iters = num_iters

    def cost(self, a, y):
        # Cross-entropy
        return np.nan_to_num(
                -y * np.log(a + 1e-24) - (1 - y) * np.log(1 - a + 1e-24))

    def fit(self, xs, ys):
        # Initializing parameters
        self.b = np.random.randn(1)[0]
        self.b_best = self.b
        self.c_best = sys.float_info.max
        self.grad_sqr_sum = 0.0
        self.iter = 0
        self.lrate = self.init_lrate
        self.w = np.random.randn(len(xs[0])) / len(xs[0])
        self.w_best = self.w.copy()
        self.xs = xs.copy()
        self.ys = ys.copy()
        # Start training!
        self.iterate(self.num_iters)

    def iterate(self, num_iters):
        start = time.time()
        for i in xrange(num_iters):
            self.iterate_()
        secs = time.time() - start
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        print 'Elapsed time %d:%02d:%02d' % (h, m, s)

    def iterate_(self):
        # Print training progress periodically.
        if self.iter % 1000 == 0:
            acc, c = self.score2(self.xs, self.ys, use_best=True)
            print 'iter %d, acc %.5f, err %.8f' % (self.iter, acc, c)

        # Outputs as_ and cost c
        as_ = self.predict(self.xs)
        c = np.mean(self.cost(as_, self.ys))

        # Keep track of the optimal model.
        if c < self.c_best:
            self.b_best = self.b
            self.c_best = c
            self.w_best = self.w.copy()

        # Gradient of bias and weights
        b_grad = np.mean(-(self.ys - as_))
        w_grad = np.dot(-(self.ys - as_), self.xs) / len(as_)

        # AdaGrad
        self.grad_sqr_sum += np.linalg.norm(w_grad) ** 2 + b_grad ** 2
        self.lrate = self.init_lrate / self.grad_sqr_sum ** 0.5

        # Update bias and weights.
        self.b -= b_grad * self.lrate
        self.w -= w_grad * self.lrate

        self.iter += 1
        return

    def predict(self, x, use_best=False):
        if use_best:
            z = np.dot(x, self.w_best) + self.b_best
        else:
            z = np.dot(x, self.w) + self.b
        return self.sigmoid(z)

    def predict2(self, xs):
        as_ = []
        for x in xs:
            a = 1 if self.predict(x, use_best=True) >= 0.5 else 0
            as_.append(a)
        return np.array(as_)

    def score(self, xs, ys, use_best=True):
        return self.score2(xs, ys, use_best)[0]

    def score2(self, xs, ys, use_best=True):
        as_ = self.predict(xs, use_best)
        c = np.mean(self.cost(as_, ys))
        acc = 0.
        for (a, y) in zip(as_, ys):
            if (a >= 0.5 and y == 1.0) or (a < 0.5 and y == 0.0):
                acc += 1
        acc /= len(as_)
        return (acc, c)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
