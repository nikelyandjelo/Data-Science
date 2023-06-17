from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
import numpy as np


class lab6():
    def __init__(self):
        tf.disable_v2_behavior()
        self.m = 40000
        self.S = np.random.rand(self.m)
        self.n = 500
        self.dm = 0
        self.dsig = 10
        self.gen()
        self.Mashtab()
        self.InitS()
        self.Rar()
        self.test()

    def gen(self):
        S = ((np.random.randn(self.n)) * self.dsig) + self.dm
        self.train = np.zeros((self.m, self.n));
        self.dteset = np.zeros((self.m, self.n));
        for j in range(self.m):
            for i in range(self.n):
                self.train[j, i] = (0.0005 * j * j) + S[i]
                self.dteset[j, i] = (0.0002 * j * j) + S[i]

        plt.plot(self.train)
        plt.plot(self.dteset)
        plt.ylabel('динаміка тестів')
        plt.show()

        print('------- train---------')
        print(self.train)
        print('------- dteset---------')
        print(self.dteset)

    def Mashtab(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.train)
        train = scaler.transform(self.train)
        dteset = scaler.transform(self.dteset)

        print('------- train_norm ---------')
        print(train)
        print('------- dteset_norm ---------')
        print(dteset)

    def InitS(self):
        self.X_train = self.train[:, 1:]
        self.y_train = self.train[:, 0]
        self.X_test = self.dteset[:, 1:]
        self.y_test = self.dteset[:, 0]
        self.n_stocks = self.X_train.shape[1]
        self.n_neurons_1 = 1024
        self.n_neurons_2 = 512
        self.n_neurons_3 = 256
        self.n_neurons_4 = 128

    def Rar(self):
        self.net = tf.InteractiveSession()

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_stocks])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None])

        sigma = 1
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()

        W_silecier_1 = tf.Variable(weight_initializer([self.n_stocks, self.n_neurons_1]))
        bias_silecier_1 = tf.Variable(bias_initializer([self.n_neurons_1]))
        W_silecier_2 = tf.Variable(weight_initializer([self.n_neurons_1, self.n_neurons_2]))
        bias_silecier_2 = tf.Variable(bias_initializer([self.n_neurons_2]))
        W_silecier_3 = tf.Variable(weight_initializer([self.n_neurons_2, self.n_neurons_3]))
        bias_silecier_3 = tf.Variable(bias_initializer([self.n_neurons_3]))
        W_silecier_4 = tf.Variable(weight_initializer([self.n_neurons_3, self.n_neurons_4]))
        bias_silecier_4 = tf.Variable(bias_initializer([self.n_neurons_4]))

        W_out = tf.Variable(weight_initializer([self.n_neurons_4, 1]))
        bias_out = tf.Variable(bias_initializer([1]))

        silecier_1 = tf.nn.relu(tf.add(tf.matmul(self.X, W_silecier_1), bias_silecier_1))
        silecier_2 = tf.nn.relu(tf.add(tf.matmul(silecier_1, W_silecier_2), bias_silecier_2))
        silecier_3 = tf.nn.relu(tf.add(tf.matmul(silecier_2, W_silecier_3), bias_silecier_3))
        silecier_4 = tf.nn.relu(tf.add(tf.matmul(silecier_3, W_silecier_4), bias_silecier_4))

        self.out = tf.transpose(tf.add(tf.matmul(silecier_4, W_out), bias_out))

        self.mse = tf.reduce_mean(tf.squared_difference(self.out, self.Y))

        self.opt = tf.train.AdamOptimizer().minimize(self.mse)

        self.net.run(tf.global_variables_initializer())

        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(self.y_test)
        self.line2, = ax1.plot(self.y_test * 0.5)
        plt.show()
        self.batch_size = 256
        self.mse_train = []
        self.mse_test = []

        self.epochs = 3

    def test(self):
        for e in range(self.epochs):

            shuffle_indices = np.random.permutation(np.arange(len(self.y_train)))
            X_train = self.X_train[shuffle_indices]
            y_train = self.y_train[shuffle_indices]
            for i in range(0, len(y_train) // self.batch_size):
                start = i * self.batch_size
                batch_x = X_train[start:start + self.batch_size]
                batch_y = y_train[start:start + self.batch_size]
                self.net.run(self.opt, feed_dict={self.X: batch_x, self.Y: batch_y})

                if np.mod(i, 50) == 0:
                    self.mse_train.append(self.net.run(self.mse, feed_dict={self.X: X_train, self.Y: y_train}))
                    self.mse_test.append(self.net.run(self.mse, feed_dict={self.X: self.X_test, self.Y: self.y_test}))
                    print('Train: ', self.mse_train[-1])
                    print('Test: ', self.mse_test[-1])

                    pred = self.net.run(self.out, feed_dict={self.X: self.X_test})
                    self.line2.set_ydata(pred)
                    plt.title(
                        'Epoch ' + str(e) + ', Batch ' + str(i))
                    plt.pause(0.5)


lab6()