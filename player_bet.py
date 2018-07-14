import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
import pandas as pd
import tensorflow as tf
from PokerBot.bet_model import SimpleNN


# TODO Close tf sessions
class SimpleBet:
    def __init__(self, call_cost, raise_cost, opponent_risk):
        self.call_cost = call_cost
        self.raise_cost = raise_cost
        self.opponent_risk = opponent_risk

    def action(self, self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise, new_vips_list):
        # must bet same amount as big blind
        win_call_cost = (curr_pot - curr_bet) * self_risk
        loss_call_cost = max_bet * (1 - self_risk)
        call_action = 'call'
        fold_action = 'fold'
        # print(win_call_cost, loss_call_cost)

        if max_bet == curr_bet:
            call_action = 'check'
            fold_action = 'check'

        if win_call_cost - loss_call_cost > self.call_cost:
            if win_call_cost > loss_call_cost + self.raise_cost:
                # big blind
                bet = (max_bet - curr_bet) + (big_blind * round((self.opponent_risk * self_risk) - (1 - self_risk)))
                # bet = (max_bet - curr_bet) * (round((self.opponent_risk * self_risk) - (1 - self_risk)))
                action = 'raise'
                bet = max(bet, max_bet - curr_bet, single_max_raise)
            else:
                bet = max_bet - curr_bet
                action = call_action
        else:
            bet = 0
            action = fold_action

        # if bet < -1:
        #     print(bet, max_bet, curr_bet)
        #     raise (ValueError, 'Cant bet negative')

        return action, bet

    def train(self, x):
        raise ValueError('Cant be trained')


class RandomBet:
    @staticmethod
    def action(self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise, new_vips_list):
        action = np.random.choice(['fold', 'call', 'raise'], 1)[0]
        if action == 'raise':
            bet = max_bet - curr_bet + np.random.randint(1, 100) * big_blind

        elif action == 'call':
            bet = max_bet - curr_bet
        else:
            bet = 0
        return action, bet

    def train(self, x):
        raise ValueError('Cant be trained')


class SimpleModelBet:
    def __init__(self, action_clf, bet_clf):
        self.clf = action_clf
        self.bet_clf = bet_clf
        # try:
        #     with open('rf_model.pkl', 'rb') as fid:
        #         self.clf = pickle.load(fid)
        # except:
        #     print('No model found - training model')
        #     self.clf = RandomForestRegressor()
        #     self.train = pd.read_csv('C:/Users/dysont/Documents/Graduate/rl/PokerBot/tournament_data.csv')
        #     self.X = self.train.drop(['Unnamed: 0', 'bet', 'player'], axis=1)
        #     self.y = self.train['bet']
        #     self.clf.fit(self.X, self.y)
        #     with open('rf_model.pkl', 'wb') as fid:
        #         pickle.dump(self.clf, fid)

    def action(self, self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise, new_vips_list):

        predict_array = pd.DataFrame({'curr_bet': curr_bet, 'curr_money': curr_money, 'curr_pot': curr_pot,
                                      'hand_lowest_money': hand_lowest_money, 'max_bet': max_bet,
                                      'remaining_players_hand': remaining_players_hand,
                                      'remaining_players_tournament': remaining_players_tournament,
                                      'self_risk': self_risk,
                                      # 'single_max_raise': single_max_raise,
                                      'table_risk': table_risk,
                                      'vips_1': new_vips_list[0],
                                      'vips_2': new_vips_list[1],
                                      'vips_3': new_vips_list[2],
                                      'vips_4': new_vips_list[3],
                                      'vips_5': new_vips_list[4],
                                      }, index=[0])
        # print(self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
        #        remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise)



        action = self.clf.predict(X=predict_array)[0]

        if action == 0:
            act_action = 'fold'
            bet = 0
        elif action == 1:
            act_action = 'call'
            bet = max_bet - curr_bet

        else:
            act_action = 'raise'
            bet = round(self.bet_clf.predict(predict_array)[0])

        return act_action, bet

    def train(self, x):
        raise ValueError('Cant be trained')


class SimpleNNBet:
    def __init__(self, number):
        # with tf.Session() as self.sess:
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(
            'C:/Users/dysont/Documents/Graduate/rl/PokerBot/runs/1531578722/checkpoints/model.ckpt-{}.meta'.format(
                number))
        saver.restore(self.sess,
                      'C:/Users/dysont/Documents/Graduate/rl/PokerBot/runs/1531578722/checkpoints/model.ckpt-{}'.format(
                          number))
        graph = tf.get_default_graph()
        self.predictions = graph.get_operation_by_name("predictions").outputs[0]
        self.input_x = graph.get_operation_by_name("input_x").outputs[0]
        self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        # correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def action(self, self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise, new_vips_list):
        predict_array = pd.DataFrame({'curr_bet': curr_bet,
                                      'curr_money': curr_money,
                                      'curr_pot': curr_pot,
                                      # 'hand_lowest_money': hand_lowest_money,
                                      # 'max_bet': max_bet,
                                      'remaining_players_tournament': remaining_players_tournament,
                                      'net_risk': self_risk - table_risk,
                                      # 'self_risk': self_risk,
                                      # 'single_max_raise': single_max_raise,
                                      # 'table_risk': table_risk,
                                      'vips_1': new_vips_list[0],
                                      'vips_2': new_vips_list[1],
                                      'vips_3': new_vips_list[2],
                                      'vips_4': new_vips_list[3],
                                      'vips_5': new_vips_list[4],
                                      }, index=[0]).values

        action = self.sess.run([self.predictions], feed_dict={
            self.input_x: predict_array,
            self.keep_prob: 1
        })[0][0]

        if action == 0:
            act_action = 'fold'
            bet = 0
        elif action == 1:
            act_action = 'call'
            bet = max_bet - curr_bet

        else:
            act_action = 'raise'
            bet = np.random.randint(2, 10) * max_bet

        # print(act_action, bet)
        return act_action, bet

    def train(self, x):
        raise ValueError('Cant be trained')


class NNRLBet:
    def __init__(self, input_len):
        self.sess = tf.Session()
        graph = tf.get_default_graph()
        self.input_len = input_len
        self.beginning_values()
        self.optimizer_cost()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

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
        n_input = self.input_len
        n_classes = 3

        self.weights = {
            'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1])),
            'out': tf.Variable(tf.random_uniform([n_hidden_1, n_classes]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_uniform([n_hidden_1])),
            'out': tf.Variable(tf.random_uniform([n_classes]))
        }

        self.keep_prob = tf.placeholder("float", name='keep_prob')
        self.x = tf.placeholder("float", [None, n_input], name='input_x')
        self.y = tf.placeholder("float", [None, n_classes], name='input_y')

    def optimizer_cost(self):
        self.predictions = self.multilayer_perceptron(self.x, self.weights, self.biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

    def prep_data(self, data):
        print(data)
        data['net_risk'] = data['self_risk'] - data['table_risk']
        data['action'] = 0
        data.loc[data.bet == 0, 'action'] = 1
        data.loc[data.bet == data.max_bet - data.curr_bet, 'action'] = 1
        data.loc[data.bet + data.curr_bet > data.max_bet, 'action'] = 2
        # data.loc[data.action == 2, 'action'] = 2 + (
        #     data.bet / data.max_bet).round()
        # data['net_risk'] = data.self_risk - data.table_risk
        # data.loc[data.action > 20, 'action'] = 20

        train_x = data[
            ['curr_bet', 'curr_money', 'curr_pot', 'remaining_players_tournament', 'net_risk',
             'vips_1', 'vips_2', 'vips_3', 'vips_4', 'vips_5', ]]

        data['0'] = 0
        data['1'] = 0
        data['2'] = 0
        data.loc[data.action == 0,'0'] = 1
        data.loc[data.action == 1,'1'] = 1
        data.loc[data.action == 2,'2'] = 1
        train_y = data[['0', '1', '2']].values
        return train_x, train_y

    def action(self, self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise, new_vips_list):
        predict_array = pd.DataFrame({'curr_bet': curr_bet,
                                      'curr_money': curr_money,
                                      'curr_pot': curr_pot,
                                      # 'hand_lowest_money': hand_lowest_money,
                                      # 'max_bet': max_bet,
                                      'remaining_players_tournament': remaining_players_tournament,
                                      'net_risk': self_risk - table_risk,
                                      # 'self_risk': self_risk,
                                      # 'single_max_raise': single_max_raise,
                                      # 'table_risk': table_risk,
                                      'vips_1': new_vips_list[0],
                                      'vips_2': new_vips_list[1],
                                      'vips_3': new_vips_list[2],
                                      'vips_4': new_vips_list[3],
                                      'vips_5': new_vips_list[4],
                                      }, index=[0]).values

        action = self.sess.run([self.predicter], feed_dict={
            self.x: predict_array,
            self.keep_prob: 1
        })[0][0]

        if action == 0:
            act_action = 'fold'
            bet = 0
        elif action == 1:
            act_action = 'call'
            bet = max_bet - curr_bet

        else:
            act_action = 'raise'
            bet = np.random.randint(2, 10) * big_blind

        # print(act_action, bet)
        return act_action, bet

    def train(self, x):
        train_x, train_y = self.prep_data(x)
        _, c = self.sess.run([self.optimizer, self.cost, ],
                             feed_dict={
                                 self.x: train_x,
                                 self.y: train_y,
                                 self.keep_prob: 0.8
                             })


if __name__ == '__main__':
    lol = 'lol'
    pkr = RandomBet
    print(pkr.action(0.239442342295, 0.256214334935, 1000, 36.0, 5, 39.0, 5, 1000, 0, 2, 2, [0, 0, 0, 0, 0]))
