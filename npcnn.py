import gzip
import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(mnist_directory):
    f = gzip.open(mnist_directory, 'rb')
    tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
    f.close()
    tr_d = [tr_d[0][:].reshape((1, 50000, 1, 28, 28)), tr_d[1]]
    va_d = [va_d[0][:].reshape((1, 10000, 1, 28, 28)), va_d[1]]
    te_d = [te_d[0][:].reshape((1, 10000, 1, 28, 28)), te_d[1]]

    #  zero-centering the data
    tr_d_mean = np.mean(tr_d[0], axis=1, keepdims=True)
    tr_d[0] -= tr_d_mean  # subtracting the mean image
    va_d[0] -= tr_d_mean
    te_d[0] -= tr_d_mean
    #  normalizing the data
    std_f = np.std(tr_d[0])
    tr_d[0] /= std_f  # dividing by standard deviation
    va_d[0] /= std_f
    te_d[0] /= std_f

    dev_data = [tr_d[0][:, :500], tr_d[1][:500]]
    return tr_d, va_d, te_d, dev_data


class Conv(object):
    def __init__(self, inp_shape, num_filters, kernel_size, padding=0):
        """
        :param inp_shape(tuple of 5 ints):
        inp_shape[0] = 1 -- additional dimension for convolution operation
        inp_shape[1] -- batch size (number of maps)
        inp_shape[2] -- number of channels in map (map depth)
        inp_shape[3] -- map heigth
        inp_shape[4] -- map width
        :param num_filters(int): num of convolution filters
        :param kernel_size(int): kernel size (h = w = s)
        :param padding(int): kernel padding
        """
        self.W = np.random.randn(num_filters, 1, inp_shape[2], kernel_size, kernel_size)\
                 * np.sqrt(2.0 / (inp_shape[2] * kernel_size ** 2))  # He init.
        self.vW = np.zeros_like(self.W)
        self.vW_prev = np.zeros_like(self.W)
        self.b = np.zeros((num_filters, 1, 1, 1))
        self.vb = np.zeros_like(self.b)
        self.vb_prev = np.zeros_like(self.b)
        self.padding = padding

    @staticmethod
    def pad(inp, pad, zeros=False):
        assert len(inp.shape) == 5, 'input must be 5d'
        assert inp.shape[3] == inp.shape[4], 'fmaps shd be square'
        if pad == 0:
            return inp
        outp = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2],
                         inp.shape[3] + 2 * pad,
                         inp.shape[4] + 2 * pad), dtype=inp.dtype)
        outp[:, :, :, pad:-pad, pad:-pad] += inp  # center
        if zeros:
            return outp
        outp[:, :, :, :pad, pad:-pad] += inp[:, :, :, :1, :]  # top
        outp[:, :, :, -pad:, pad:-pad] += inp[:, :, :, -1:, :]  # bottom
        outp[:, :, :, pad:-pad, :pad] += inp[:, :, :, :, :1]  # left
        outp[:, :, :, pad:-pad, -pad:] += inp[:, :, :, :, -1:]  # right
        outp[:, :, :, :pad, :pad] += inp[:, :, :, :1, :1]  # top-left
        outp[:, :, :, :pad, -pad:] += inp[:, :, :, :1, -1:]  # top-right
        outp[:, :, :, -pad:, :pad] += inp[:, :, :, -1:, :1]  # bottom-left
        outp[:, :, :, -pad:, -pad:] += inp[:, :, :, -1:, -1:]  # bottom-right
        return outp

    def f_prop(self, inp, p=0.5):
        inp_resized = self.pad(inp, self.padding)

        out = np.zeros((self.W.shape[0], inp.shape[1],
                        inp.shape[3] - self.W.shape[3] + 2 * self.padding + 1,
                        inp.shape[4] - self.W.shape[4] + 2 * self.padding + 1), dtype=float)
        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                out[:, :, i:i + 1, j:j + 1] = (inp_resized[:, :, :, i:i + self.W.shape[3], j:j + self.W.shape[4]] *
                                               self.W).sum(axis=(3, 4), keepdims=True).sum(axis=2)
        out += self.b
        out = np.stack([out.swapaxes(0, 1)])
        self.cache = (inp,)
        return out

    def b_prop(self, dout, reg=0.0):
        dout_resized = self.pad(dout, self.W.shape[3] - self.padding - 1, zeros=True)
        dinp = np.zeros_like(self.cache[0])[0].swapaxes(0, 1)
        W_bp = np.rot90(self.W, 2, axes=(3, 4)).swapaxes(0, 2)
        for i in range(dinp.shape[2]):
            for j in range(dinp.shape[3]):
                dinp[:, :, i:i + 1, j:j + 1] =\
                    (dout_resized[:, :, :, i:i + W_bp.shape[3], j:j + W_bp.shape[4]] * W_bp)\
                    .sum(axis=(3, 4), keepdims=True).sum(axis=2)
        dinp = np.stack([dinp.swapaxes(0, 1)])

        self.dW = np.zeros_like(self.W)
        self.dW = self.dW.swapaxes(0, 1)[0]
        self.db = np.zeros_like(self.b)
        inp_resized = self.pad(self.cache[0],
                               int((self.W.shape[3] - self.cache[0].shape[3] + dout.shape[3] - 1) / 2),
                               zeros=True)
        inp_resized = inp_resized.swapaxes(1, 2)
        dout = np.rot90(dout, 2, (3, 4))
        dout = dout.swapaxes(1, 2)
        self.db[:] = dout[0].sum(axis=(1, 2, 3), keepdims=True)
        dout = dout.swapaxes(0, 1)
        for i in range(self.dW.shape[2]):
            for j in range(self.dW.shape[3]):
                self.dW[:, :, i:i + 1, j:j + 1] =\
                    (inp_resized[:, :, :, i:i + dout.shape[3], j:j + dout.shape[4]] * dout)\
                    .sum(axis=(3, 4), keepdims=True).sum(axis=2)
        self.dW = np.stack([self.dW]).swapaxes(0, 1)
        self.dW += reg * self.W
        del self.cache
        return dinp

    def upd(self, eta, mu):
        # Nesterov Accelerated Gradient (NAG)
        self.vW_prev = self.vW
        self.vW = mu * self.vW - eta * self.dW
        self.W += -mu * self.vW_prev + (1 + mu) * self.vW

        self.vb_prev = self.vb
        self.vb = mu * self.vb - eta * self.db
        self.b += -mu * self.vb_prev + (1 + mu) * self.vb

    def fp(self, inp):
        pass


class ReLU(object):
    def f_prop(self, inp, p=0.5):
        self.cache = (inp,)
        return np.maximum(0, inp)

    def b_prop(self, dout, reg=0.0):
        dinp = np.where(self.cache[0] > 0, dout, 0)
        del self.cache
        return dinp

    def upd(self, eta, mu):
        pass

    def fp(self, inp):
        return np.maximum(0, inp)


class MaxPool(object):
    def f_prop(self, inp, p=0.5):
        assert len(inp.shape) == 5, 'input must be 5d'
        assert inp.shape[3] % 2 == 0, 'h % 2 != 0'
        assert inp.shape[4] % 2 == 0, 'w % 2 != 0'

        out = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2],
                        int(inp.shape[3] / 2), int(inp.shape[4] / 2)), dtype=inp.dtype)
        maxmap = np.zeros(inp.shape, dtype=int)
        for i in range(out.shape[3]):
            for j in range(out.shape[4]):
                out[:, :, :, i:i + 1, j:j + 1] = inp[:, :, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2].reshape(
                    inp.shape[:-2] + (1, 4)
                ).max(axis=-1, keepdims=True)
                maxmap[:, :, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2] =\
                    inp[:, :, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2] == out[:, :, :, i:i + 1, j:j + 1]

        self.cache = (inp.shape, maxmap)
        return out

    def b_prop(self, dout, reg=0.0):
        assert len(self.cache[0]) == 5, 'input must be 5d'
        assert self.cache[0][3] % 2 == 0, 'h % 2 != 0'
        assert self.cache[0][4] % 2 == 0, 'w % 2 != 0'

        inp_shape, maxmap = self.cache
        dinp = np.zeros(inp_shape, dtype=float)

        for i in range(dout.shape[3]):
            for j in range(dout.shape[4]):
                dinp[:, :, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2] =\
                    maxmap[:, :, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2] * dout[:, :, :, i:i + 1, j:j + 1]

        del self.cache
        return dinp

    def upd(self, eta, mu):
        pass

    def fp(self, inp):
        pass


class FullyConnected(object):
    def __init__(self, inp_shape, hidden, classes):
        inp_size = inp_shape[2] * inp_shape[3] * inp_shape[4]
        layers = [inp_size] + hidden + [classes]
        self.W = [np.random.randn(prev, nxt) * np.sqrt(2.0 / prev) for prev, nxt
                  in zip(layers[:-1], layers[1:])]  # He init.
        self.b = [np.zeros((1, layer), dtype=float) for layer in layers[1:]]
        self.vb = [0 for b in self.b]
        self.vb_prev = [0 for b in self.b]
        self.vW = [0 for W in self.W]
        self.vW_prev = [0 for W in self.W]

    def f_prop(self, X, p=0.5):
        # Forward pass
        # concatenate
        N = X.shape[1]
        inp = X.reshape((N, -1))

        # evaluate class scores
        z = np.maximum(0, inp.dot(self.W[0]) + self.b[0])

        if p:
            # dropout
            u = (np.random.rand(*z.shape) < p) / p
            z *= u

        zs = [z]
        for w, b in zip(self.W[1:-1], self.b[1:-1]):
            z = np.maximum(0, z.dot(w) + b)

            if p:
                # dropout
                u = (np.random.rand(*z.shape) < p) / p
                z *= u

            zs.append(z)
        scores = z.dot(self.W[-1]) + self.b[-1]

        # Compute the loss (Softmax classifier loss)

        # Compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.cache = (X.shape, inp, zs, probs)

        return scores

    def b_prop(self, y, reg=0.0):
        X_shape, inp, zs, probs = self.cache
        del self.cache

        # Compute the loss: average cross-entropy loss and regularization
        N = len(y)
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.0
        for W in self.W:
            reg_loss += 0.5 * reg * np.sum(W * W)

        loss = data_loss + reg_loss

        dscores = np.zeros_like(probs)
        dscores[:] = probs
        dscores[range(N), y] -= 1
        dscores /= N

        self.dW = [np.zeros_like(W) for W in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

        self.dW[-1] = zs[-1].T.dot(dscores)
        self.db[-1] = np.sum(dscores, axis=0, keepdims=True)
        dz = dscores.dot(self.W[-1].T)
        dz[zs[-1] <= 0] = 0
        for i in range(len(self.W) - 2, 0, -1):
            self.dW[i] = zs[i - 1].T.dot(dz)
            self.db[i] = np.sum(dz, axis=0, keepdims=True)

            dz = dz.dot(self.W[i].T)
            dz[zs[i - 1] <= 0] = 0
        self.dW[0] = inp.T.dot(dz)
        self.db[0] = np.sum(dz, axis=0, keepdims=True)

        # regularization gradient
        for i in range(len(self.dW)):
            self.dW[i] += reg * self.W[i]
        dinp = dz.dot(self.W[0].T).reshape(X_shape)
        return dinp, loss

    def upd(self, eta, mu):

        # NAG
        self.vW_prev = [vW for vW in self.vW]
        self.vW = [mu * vW - eta * dW for vW, dW in zip(self.vW, self.dW)]
        self.W = [W + (-mu * vW_prev + (1 + mu) * vW) for W, vW_prev, vW in zip(self.W, self.vW_prev, self.vW)]

        self.vb_prev = [vb for vb in self.vb]
        self.vb = [mu * vb - eta * db for vb, db in zip(self.vb, self.db)]
        self.b = [b + (-mu * vb_prev + (1 + mu) * vb) for b, vb_prev, vb in zip(self.b, self.vb_prev, self.vb)]

    def fp(self):
        pass


class Net(object):
    def __init__(self, layers):
        """
        e.g.:
        layers = {
            'conv': {
                'inp_shape': (),
                'num_filters': 0,
                'kernel_size': 0,
                'padding': 0
            },
            'relu': None,
            'pool': None,
            'conv': {
                'inp_shape': (),
                'num_filters': 0,
                'kernel_size': 0,
                'padding': 0
            },
            'relu': None,
            'conv': {
                'inp_shape': (),
                'num_filters': 0,
                'kernel_size': 0,
                'padding': 0
            },
            'relu': None,
            'pool': None,
            'full': {
                'inp_shape': (),
                'hidden': [],
                'classes': 0
            }
        }
        """
        self.layers = np.array(layers)

    def f_prop(self, inp, p=0.5):
        for layer in self.layers:
            inp = layer.f_prop(inp, p=p)
        scores = inp
        return scores

    def b_prop(self, y, reg=0.0):
        dinp, loss = self.layers[-1].b_prop(y, reg)
        for layer in self.layers[-2::-1]:
            dinp = layer.b_prop(dinp, reg)
        return loss

    def upd(self, eta, mu):
        for layer in self.layers:
            layer.upd(eta, mu)

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, mu=0.5,
              learning_rate_decay=0.95, reg=5e-6, epochs=1, p=0.5,
              batch_size=200, verbose=False, shuffle=False):


        X_shfld = np.zeros_like(X)
        y_shfld = np.zeros_like(y)
        X_val_shfld = np.zeros_like(X_val)
        y_val_shfld = np.zeros_like(y_val)

        num_train = X.shape[1]
        num_val = X_val.shape[1]

        mask = np.arange(num_train)
        mask_val = np.arange(num_val)

        iterations_per_epoch = int(num_train / batch_size)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(epochs):
            print('\n\n--------------|EPOCH %i|--------' % (epoch))
            if shuffle:
                np.random.shuffle(mask)
                np.random.shuffle(mask_val)
            X_shfld[:] = X[:, mask]
            y_shfld[:] = y[mask]
            X_val_shfld[:] = X_val[:, mask_val]
            y_val_shfld[:] = y_val[mask_val]
            tic = time()
            for it in range(iterations_per_epoch):
                _ = self.f_prop(X_shfld[:, it * batch_size: (it + 1) * batch_size], p=p)
                loss = self.b_prop(y_shfld[it * batch_size: (it + 1) * batch_size], reg=reg)

                self.upd(learning_rate, mu)

                loss_history.append(loss)

                if verbose:
                    print('------|%i | loss = %f' % (((it + 1) * 100) / iterations_per_epoch, loss))
            toc = time()

            print('---|epoch %i finished in %f seconds' % (epoch, toc - tic))

            train_acc = (self.f_prop(X_shfld, p=None).argmax(axis=1) == y_shfld).mean()
            val_acc = (self.f_prop(X_val_shfld, p=None).argmax(axis=1) == y_val_shfld).mean()

            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            print('------|train_acc = %f' % (train_acc))
            print('--------|val_acc = %f' % (val_acc))
            learning_rate *= learning_rate_decay

        return loss_history, train_acc_history, val_acc_history

    def load_model(self):
        pass

    def save_model(self, filename):
        pass

    def vis_grid(self):
        pass

train_data, val_data, test_data, dev_data = load_mnist('data/mnist.pkl.gz')

conv0 = Conv(dev_data[0].shape, 4, 5)
relu0 = ReLU()
pool0 = MaxPool()
tmp = pool0.f_prop(conv0.f_prop(dev_data[0]))
conv1 = Conv(tmp.shape, 8, 3, 1)
relu1 = ReLU()
tmp = conv1.f_prop(tmp)
conv2 = Conv(tmp.shape, 16, 3)
relu2 = ReLU()
pool2 = MaxPool()
tmp = pool2.f_prop(conv2.f_prop(tmp))
fc = FullyConnected(tmp.shape, [160], 10)
arch = [conv0, relu0, pool0,
        conv1, relu1,
        conv2, relu2, pool2,
        fc]
convnet = Net(arch)
loss_h, tr_acc_h, va_acc_h = convnet.train(train_data[0], train_data[1],
                                           val_data[0], val_data[1],
                                           learning_rate=1e-2, mu=0.5,
                                           learning_rate_decay=0.99, reg=1e-3,
                                           epochs=5, p=0.7, batch_size=50,
                                           verbose=False, shuffle=True)
plot(loss_h, tr_acc_h, va_acc_h)

_, _, _ = convnet.train(test_data[0][:, :1], test_data[1][:1],
                        test_data[0][:, 1:], test_data[1][1:],
                                           learning_rate=1e-2, mu=0.5,
                                           learning_rate_decay=0.99, reg=1e-3,
                                           epochs=5, p=0.7, batch_size=1,
                                           verbose=False, shuffle=True)