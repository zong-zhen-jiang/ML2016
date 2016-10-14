import numpy as np
import random


class NeuralNetwork(object):

    def __init__(self, batch_size, data_test, data_train, sizes, step_size,
            te_err_intv, tr_err_intv):
        self.batch_size = batch_size
        self.data_test = data_test
        self.data_train = data_train
        self.iter_idx = 0
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.step_size = step_size
        self.te_err_intv = te_err_intv
        self.tr_err_intv = tr_err_intv

        self.x = np.random.randn(sizes[0], 1)
        self.ws = [np.random.randn(ny, nx)
                for (nx, ny) in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.random.randn(ny, 1) for ny in sizes[1:]]
        self.zs = [np.random.randn(ny, 1) for ny in sizes[1:]]
        self.ys = [np.random.randn(ny, 1) for ny in sizes[1:]]

        random.shuffle(data_train)
        self.batches = [data_train[i:i+batch_size]
                for i in xrange(0, len(data_train), batch_size)]
        self.batch_idx = 0

    def act_func(self, z):
        # TODO Implement other functions.
        #return z
        return 1. / (1. + np.exp(-z))
        #return 1. / (1. + np.exp(-z * 0.1))

    def act_func_deri(self, z):
        # TODO Implement other function derivatives.
        #return 1.0
        return self.act_func(z) * (1. - self.act_func(z))
        #return self.act_func(z * 0.1) * (1. - self.act_func(z * 0.1)) * 0.1

    def cost_func(self, y, label):
        return (y - label) ** 2

    def cost_func_deri(self, y, label):
        return 2 * (y - label)

    def error(self, data=None):
        if data == None:
            data = self.batches[self.batch_idx - 1]
        error = np.zeros(self.ys[-1].shape)
        acc = 0.0
        for (x, l) in data:
            y = self.forward(x)
            error += self.cost_func(y, l)
            acc += int(l[np.argmax(y)] > 0.9)
        return (error / len(data), acc / len(data))
    
    def error2(self, data=None):
        if data == None:
            data = self.batches[self.batch_idx - 1]
        error = np.zeros(self.ys[-1].shape)
        acc = 0.0
        for (x, l) in data:
            self.forward(x)
            l = np.log(l / (1. - l))
            error += self.cost_func(self.zs[-1], l)
            #acc += int(l[np.argmax(y)] > 0.9)
        return (error / len(data), acc / len(data))

    def forward(self, x):
        self.x = x
        for i in xrange(self.num_layers - 1):
            self.zs[i] = np.dot(self.ws[i], x) + self.bs[i]
            self.ys[i] = self.act_func(self.zs[i])
            x = self.ys[i]
        return self.ys[-1]

    def iterate(self, num_iters=1):
        for i in xrange(num_iters):
            self.iterate_()

    def iterate_(self):
        sum_grad_w = [np.zeros(w.shape, dtype=long) for w in self.ws]
        sum_grad_b = [np.zeros(b.shape, dtype=long) for b in self.bs]

        batch = self.batches[self.batch_idx]
        for (x, l) in batch:
            self.forward(x)
            (grad_w, grad_b) = self.backprop(l)
            sum_grad_w = [sgw + gw for (sgw, gw) in zip(sum_grad_w, grad_w)]
            sum_grad_b = [sgb + gb for (sgb, gb) in zip(sum_grad_b, grad_b)]

        self.ws = [w - self.step_size * ((sgw + 0.1 * w) / len(batch))
                for (w, sgw) in zip(self.ws, sum_grad_w)]
        self.bs = [b - self.step_size * (sgb / len(batch))
                for (b, sgb) in zip(self.bs, sum_grad_b)]

        if self.iter_idx % self.tr_err_intv == 0:
            print 'Train err at iter #{}: {}'.format(
                    self.iter_idx, self.error2(self.data_train))
        if self.iter_idx % self.te_err_intv == 0:
            print 'Valid err at iter #{}: {}'.format(
                    self.iter_idx, self.error2(self.data_test))

        self.iter_idx += 1
        self.batch_idx += 1
        if self.batch_idx >= len(self.batches):
            random.shuffle(self.data_train)
            self.batches = [self.data_train[i:i+self.batch_size]
                    for i in xrange(0, len(self.data_train), self.batch_size)]
            self.batch_idx = 0

    def backprop(self, label):
        nabla_w = [np.zeros(w.shape) for w in self.ws]
        nabla_b = [np.zeros(b.shape) for b in self.bs]

        delta = self.cost_func_deri(self.ys[-1], label) * \
                self.act_func_deri(self.zs[-1])
        nabla_w[-1] = np.dot(delta, self.ys[-2].transpose())
        nabla_b[-1] = delta

        for l in xrange(2, self.num_layers):
            prev_y = self.x if l == self.num_layers - 1 else self.ys[-l - 1]
            delta = np.dot(self.ws[-l + 1].transpose(), delta) * \
                    self.act_func_deri(self.zs[-l])
            nabla_w[-l] = np.dot(delta, prev_y.transpose())
            nabla_b[-l] = delta

        return (nabla_w, nabla_b)

