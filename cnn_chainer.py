import datetime
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import cm

from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer.datasets import tuple_dataset
from sklearn.datasets import *
from chainer.training import extensions
from sklearn.datasets import fetch_mldata

class CNN(Chain):
    def __init__(self, n_out):
        super(CNN, self).__init__()
        with self.init_scope():
            conv1 = L.Convolution2D(None, 20, 5)
            conv2 = L.Convolution2D(20, 50, 5)
            l1 = L.Linear(800, 500)
            l2 = L.Linear(500, 500)
            l3 = L.Linear(500, n_out, initialW=np.zeros((n_out, 500), dtype=np.float32))

    def __call__(self, x, t=None, train=False):
        # 順伝搬の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return 計算した損失 or 予測したラベル

        x = Variable(x)
        if train:
            t = Variable(t)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h

class LeNet(Chain):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 6, 5, stride=1)
            self.conv2 = L.Convolution2D(None, 16, 5, stride=1)
            self.fc3 = L.Linear(None, 120)
            self.fc4 = L.Linear(None, 64)
            self.fc5 = L.Linear(None, n_out)

    def __call__(self, x, t=None, train=False):
        # 順伝搬の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return 計算した損失 or 予測したラベル

        x = Variable(x)
        if train:
            t = Variable(t)
        h = F.max_pooling_2d(F.local_response_normalization(F.sigmoid(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.sigmoid(self.conv2(h))), 2, stride=2)
        h = F.sigmoid(self.fc3(h))
        h = F.sigmoid(self.fc4(h))
        h = self.fc5(h)
        return h

class AlexNet(Chain):
    def __init__(self, n_out):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=2)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, n_out)

    def __call__(self, x, t=None, train=False):
        # 順伝搬の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return 計算した損失 or 予測したラベル

        x = Variable(x)
        if train:
            t = Variable(t)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h

class VGG16(Chain):
    def __init__(self, n_out):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7 = L.Convolution2D(None, 256, 3, stride=1, pad=1)

            self.conv8 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv9 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv11 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv13 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc14 = L.Linear(None, 4096)
            self.fc15 = L.Linear(None, 4096)
            self.fc16 = L.Linear(None, n_out)

    def __call__(self, x, t=None, train=False):
        # 順伝搬の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return 計算した損失 or 予測したラベル

        x = Variable(x)
        if train:
              t = Variable(t)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv7(h))), 2, stride=2)

        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv10(h))), 2, stride=2)

        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv13(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc14(h)))
        h = F.dropout(F.relu(self.fc15(h)))
        h = self.fc16(h)
        return h


class VGG19(Chain):
    def __init__(self, n_out):
        super(VGG19, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv8 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv9 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv11 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv13 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv14 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv15 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv16 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc17 = L.Linear(None, 4096)
            self.fc18 = L.Linear(None, 4096)
            self.fc19 = L.Linear(None, n_out)

    def __call__(self, x, t=None, train=False):
        # 順伝搬の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return 計算した損失 or 予測したラベル

        x = Variable(x)
        if train:
            t = Variable(t)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv8(h))), 2, stride=2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.relu(self.conv11(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv12(h))), 2, stride=2)

        h = F.relu(self.conv13(h))
        h = F.relu(self.conv14(h))
        h = F.relu(self.conv15(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv16(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc17(h)))
        h = F.dropout(F.relu(self.fc18(h)))
        h = self.fc19(h)

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

# 学習

EPOCH_NUM = 5
BATCH_SIZE = 1000

mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(np.float32) # 画像データ 784*70000 [[0-255, 0-255, ...], [0-255, 0-255, ...],...]
mnist.data /= 255 # 0-1に正規化
mnist.target = mnist.target.astype(np.int32) #ラベルデータ 70000

#教師データを変換
N = 60000
train_x, test_x = np.split(mnist.data,   [N])
train_t, test_t = np.split(mnist.target, [N])
train_x = train_x.reshape((len(train_x), 1, 28, 28)) # N, channel, height, width
test_x = test_x.reshape((len(test_x), 1, 28, 28))
train = tuple_dataset.TupleDataset(train_x, train_t)
test = tuple_dataset.TupleDataset(test_x, test_t)
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

#モデルの定義
model = L.Classifier(LeNet(10))
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()