import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tensorflow as tf
import numpy as np
import os
import time


class RFModel:
    def __init__(self, filepath, sample=None):
        self.train_data = pd.read_csv(filepath)
        print(len(self.train_data))
        self.prepped = False
        self.sample = sample

    def prep_data(self):
        if self.prepped:
            return self.train_data
        if self.sample is not None:
            self.train_data = self.train_data.sample(self.sample)
        self.train_data['action'] = 0
        self.train_data.loc[self.train_data.bet == 0, 'action'] = 1
        self.train_data.loc[self.train_data.bet == self.train_data.max_bet - self.train_data.curr_bet, 'action'] = 1
        self.train_data.loc[self.train_data.bet + self.train_data.curr_bet > self.train_data.max_bet, 'action'] = 2
        return self.train_data

    def action_model(self):
        action_clf = RandomForestClassifier()
        data = self.prep_data()
        X = data.drop(['Unnamed: 0', 'bet', 'player', 'action', 'single_max_raise'], axis=1)
        y = data.action
        action_clf.fit(X, y)
        return action_clf

    def bet_model(self):
        bet_clf = RandomForestRegressor()
        data = self.prep_data()
        bet_data = data[data.action == 2]
        bet_X = bet_data.drop(['Unnamed: 0', 'bet', 'player', 'action', 'single_max_raise'], axis=1)
        bet_y = bet_data['bet']
        bet_clf.fit(bet_X, bet_y)
        return bet_clf


class SimpleNN:
    def __init__(self, filepath, display_step=10, batch_size=32):
        self.train_data = pd.read_csv(filepath)
        self.epoch_len = len(self.train_data)
        self.display_step = display_step
        self.batch_size = batch_size
        self.weights = None
        self.biases = None
        self.keep_prob = None
        self.x = None
        self.y = None
        self.optimizer = None
        self.cost = None
        self.train_x = None
        self.train_y = None
        self.timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", self.timestamp))
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model.ckpt")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = None

    def prep_data(self):
        self.train_data['action'] = 0
        self.train_data.loc[self.train_data.bet == 0, 'action'] = 1
        self.train_data.loc[self.train_data.bet == self.train_data.max_bet - self.train_data.curr_bet, 'action'] = 1
        self.train_data.loc[self.train_data.bet + self.train_data.curr_bet > self.train_data.max_bet, 'action'] = 2
        # self.train_data.loc[self.train_data.action == 2, 'action'] = 2 + (
        #     self.train_data.bet / self.train_data.max_bet).round()
        # self.train_data.loc[self.train_data.action > 20, 'action'] = 20
        self.train_data['net_risk'] = self.train_data.self_risk - self.train_data.table_risk
        print(self.train_data.groupby('action').curr_money.count())
        self.train_data = pd.concat([self.train_data[self.train_data.action == 0].sample(500000),
                                     self.train_data[self.train_data.action == 1].sample(500000),
                                     self.train_data[self.train_data.action == 2].sample(500000),])
        # self.train_x = self.train_data.drop(['Unnamed: 0', 'bet', 'player', 'action', 'single_max_raise', ],
        #                                     axis=1).values
        self.train_x = self.train_data[
            ['curr_bet', 'curr_money', 'curr_pot', 'remaining_players_tournament', 'net_risk',
             'vips_1', 'vips_2', 'vips_3', 'vips_4', 'vips_5', ]]
        print(self.train_data.groupby('action').curr_money.count())
        self.train_y = pd.get_dummies(self.train_data.action).values

    def multilayer_perceptron(self, x, weights, biases, keep_prob):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        self.predicter = tf.argmax(out_layer, 1, name='predictions')
        self.predict_proba = tf.nn.softmax(out_layer, name='predict_proba')
        return out_layer

    def beginning_values(self):
        n_hidden_1 = 38
        n_input = self.train_x.shape[1]
        print(self.train_y.shape)
        n_classes = self.train_y.shape[1]

        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        self.keep_prob = tf.placeholder("float", name='keep_prob')
        self.x = tf.placeholder("float", [None, n_input], name='input_x')
        self.y = tf.placeholder("float", [None, n_classes], name='input_y')

    def optimizer_cost(self):
        self.predictions = self.multilayer_perceptron(self.x, self.weights, self.biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

    def train(self, epochs=500):
        with tf.Graph().as_default():
            self.prep_data()
            self.beginning_values()
            self.optimizer_cost()
            with tf.Session() as sess:
                self.saver = tf.train.Saver(max_to_keep=5)
                sess.run(tf.global_variables_initializer())

                for epoch in range(epochs):
                    avg_cost = 0.0
                    total_batch = int(len(self.train_x) / self.batch_size)
                    x_batches = np.array_split(self.train_x, total_batch)
                    y_batches = np.array_split(self.train_y, total_batch)
                    print(total_batch)
                    for i in range(total_batch):
                        batch_x, batch_y = x_batches[i], y_batches[i]
                        _, c, preds = sess.run([self.optimizer, self.cost, self.biases],
                                               feed_dict={
                                                   self.x: batch_x,
                                                   self.y: batch_y,
                                                   self.keep_prob: 0.8
                                               })
                        avg_cost += c
                        # print(c)
                        # print(avg_cost, type(avg_cost))
                        if np.isnan(np.min(c)):
                            print(i)
                            raise ValueError

                            # print('batch : {}, cost : {}'.format(i, c))
                    if epoch % 5 == 0:
                        self.saver.save(sess, self.checkpoint_prefix, global_step=epoch)
                    print("Epoch:", '%04d' % (epoch + 1), "cost=",
                          "{:.9f}".format(avg_cost / self.epoch_len))
                print("Optimization Finished!")

    def predict(self, number):

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                'C:/Users/dysont/Documents/Graduate/rl/PokerBot/runs/1530737332/checkpoints/model.ckpt-{}.meta'.format(
                    number))
            saver.restore(sess,
                          'C:/Users/dysont/Documents/Graduate/rl/PokerBot/runs/1530737332/checkpoints/model.ckpt-{}'.format(
                              number))
            graph = tf.get_default_graph()
            predictions = graph.get_operation_by_name("predictions").outputs[0]
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

            curr_x = self.train_x[0:100]
            preds = sess.run([predictions], feed_dict={
                input_x: curr_x,
                keep_prob: 1
            })
            print(preds[0])


if __name__ == '__main__':
    nn_obj = SimpleNN('C:/Users/dysont/Documents/Graduate/rl/PokerBot/tournament_data_vips.csv')
    # nn_obj.prep_data()
    # print(nn_obj.train_x)
    nn_obj.train(epochs=100)
    # print(nn_obj.predict(95))
    # action_nn = SimpleNNAction(95)

    # print(action_nn.action(0.239442342295, 0.256214334935, 1000, 36.0, 5, 39.0, 5, 1000, 0, 2, 2))
