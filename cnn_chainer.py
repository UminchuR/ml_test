import datetime
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import cm

from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

from sklearn.datasets import fetch_mldata

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            conv1 = L.Convolution2D(None, 20, 5)
            conv2 = L.Convolution2D(20, 50, 5)
            l1 = L.Linear(800, 500)
            l2 = L.Linear(500, 500)
            l3 = L.Linear(500, 10, initialW=np.zeros((10, 500), dtype=np.float32))

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
        y = F.softmax(self.l3(h))

        if train:
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            return loss, accuracy
        else:
            return np.argmax(y.data)

    def reset(self):
        # 勾配の初期化
        self.cleargrads()

class LeNet(Chain):
    def __init__(self):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 6, 5, stride=1)
            self.conv2 = L.Convolution2D(None, 16, 5, stride=1)
            self.fc3 = L.Linear(None, 120)
            self.fc4 = L.Linear(None, 64)
            self.fc5 = L.Linear(None, 10)

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
        y = F.softmax(self.fc5(h))

        if train:
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            return loss, accuracy
        else:
            return np.argmax(y.data)

    def reset(self):
        # 勾配の初期化
        self.cleargrads()

class AlexNet(Chain):
    def __init__(self):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=2)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 10)

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
        y = F.softmax(self.fc8(h))

        if train:
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            return loss, accuracy
        else:
            return np.argmax(y.data)

    def reset(self):
        # 勾配の初期化
        self.cleargrads()


class VGG16(Chain):
    def __init__(self):
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
            self.fc16 = L.Linear(None, 10)

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
        y = F.softmax(self.fc16(h))

        if train:
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            return loss, accuracy
        else:
            return np.argmax(y.data)

    def reset(self):
        # 勾配の初期化
        self.cleargrads()


class VGG19(Chain):
    def __init__(self):
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
            self.fc19 = L.Linear(None, 10)

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
        h = F.softmax(self.fc19(h))

        if train:
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            return loss, accuracy
        else:
            return np.argmax(y.data)

    def reset(self):
        # 勾配の初期化
        self.cleargrads()

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

#モデルの定義
model = AlexNet()
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習開始
print("Train")
st = datetime.datetime.now()
for epoch in range(EPOCH_NUM):
    # ミニバッチ学習
    perm = np.random.permutation(N) #ランダムな性数列リストを取得
    total_loss = 0
    total_accuracy = 0
    for i in range(0, N, BATCH_SIZE):
        x = train_x[perm[i:i+BATCH_SIZE]]
        t = train_t[perm[i:i+BATCH_SIZE]]
        model.reset()
        (loss, accuracy) = model(x=x, t=t, train=True)
        loss.backward()
        loss.unchain_backward()
        total_loss += loss.data
        total_accuracy += accuracy.data
        optimizer.update()
    ed = datetime.datetime.now()
    print("epoch:\t{}\ttotal loss:\t{}\tmean accuracy\t{}\ttime\t{}".format(epoch+1, total_loss, total_accuracy/(N/BATCH_SIZE), ed-st))
    st = datetime.datetime.now()

# 予測

print("\nPredict")
def predict(model, x):
    y = model(x=np.array([x], dtype="float32"), train=False)
    plt.figure(figsize=(1,1))
    plt.imshow(x[0], cmap=cm.gray_r)
    plt.show()
    print("y:\t{}\n".format(y))

idx = np.random.choice((70000-N), 10)
for i in idx:
    predict(model, test_x[i])