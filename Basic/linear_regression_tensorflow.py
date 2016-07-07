import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.core.multiarray import ndarray
class LinearRegression:
    def __init__(self, train_X: ndarray, train_Y: ndarray, learning_rate=0.001, training_epochs=100):
        self.train_X = train_X
        self.train_Y = train_Y
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs

    def fit(self):
        x = tf.placeholder("float")
        y = tf.placeholder("float")
        a = tf.Variable(1.0, name="weight")
        b = tf.Variable(1.0, name="bias")

        pred = tf.mul(x, a) + b

        cost = tf.reduce_mean(tf.abs(pred - y))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                for i, out in zip(self.train_X, self.train_Y):
                    sess.run(optimizer, feed_dict={x: i, y: out})
                    # c=sess.run(cost)
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "W=", sess.run(a), "b=", sess.run(b))
            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={x: self.train_X, y: self.train_Y})
            print("Training cost=", training_cost, "a=", sess.run(a), "b=", sess.run(b), '\n')
            return sess.run(a), sess.run(b)


def visualize(a, b, train_X: ndarray, train_Y: ndarray):
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, train_Y)
    plt.plot(train_X, a * train_X + b, label='Fitted line')
    plt.scatter(train_X, train_Y)
    # plt.plot(train_X, sess.run(a) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


def data_maker(num=80):
    X = np.arange(0, num, dtype=np.float32)
    Y = np.float32(np.ceil(5 * (np.sin(X) + X / 5)))
    return X, Y

if __name__ == "__main__":
    data = data_maker(5)
    # print(data)
    # reduce_dimension_and_plot(    print(data))
    # plot(*data_maker())
    regression = LinearRegression(*data_maker())
    visualize(*(regression.fit()+data_maker()))