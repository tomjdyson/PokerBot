import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LeakyReLU, Activation
from keras.optimizers import SGD, Adam
from keras import initializers
import pandas as pd


# TODO Set high initial estimates for Q-values - optimistic initialisation
# TODO Soft selection, sometimes select non highest q value
# TODO Change checking - doesnt work on new hand
# TODO Add in ar model again
# TODO Try increased net size
# TODO Fold reward > 0
# TODO Randomise money

class NFSP:
    def __init__(self, bellman_value=0.95, rl_buffer_size=40000, sl_buffer_size=100000, batch_size=1028, n_hidden=64,
                 state_size=11, anticipatory=1, epsilon=0.08):
        self.card_map = self.create_card_map()
        self.state_memory = []
        self.rl_memory = []
        self.bellman_value = bellman_value
        self.rl_buffer_size = rl_buffer_size
        self.sl_buffer_size = sl_buffer_size
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.s_dim = (state_size,)
        self.state_size = (1, state_size)
        self.br_model = self._build_best_response_model()
        self.q_br_model = self._build_best_response_model()
        self.ar_model = self._build_average_response_model()
        self.sl_memory = []
        self.anticipatory = anticipatory
        self.epsilon = epsilon

    @staticmethod
    def create_cards():
        numbers = list(range(1, 14))
        suits = ['c', 'h', 's', 'd']
        cards = [(i, j) for i in numbers for j in suits]
        return cards

    # def _build_best_response_model(self):
    #     initializer = initializers.uniform(minval=0.5, maxval=1)
    #     input_ = Input(shape=self.s_dim, name='input')
    #     hidden = Dense(self.n_hidden, activation='relu')(input_)
    #     # hidden = Dense(self.n_hidden, activation='relu', kernel_initializer=initializer)(input_)
    #     out = Dense(3, activation='tanh')(hidden)
    #     model = Model(inputs=input_, outputs=out, name="br-model")
    #     model.compile(loss='mean_squared_error', optimizer=SGD(0.1), metrics=['accuracy', 'mse'])
    #     return model
    #
    def _build_best_response_model(self):
        model = Sequential([
            Dense(self.n_hidden, input_shape=self.s_dim),
            LeakyReLU(),
            Dense(3),
            LeakyReLU(),
        ])
        model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy', 'mse'])
        return model

    def _build_average_response_model(self):

        initializer = initializers.uniform(minval=0.5, maxval=1)
        input_ = Input(shape=self.s_dim, name='input')
        # todo potential but needs more understanding
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        # hidden = Dense(self.n_hidden, activation='relu', kernel_initializer=initializer)(input_)
        out = Dense(3, activation='softmax')(hidden)
        model = Model(inputs=input_, outputs=out, name="br-model")
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy', 'mse'])
        return model

    def create_card_map(self):
        cards = self.create_cards()
        return {cards[i]: i for i in range(len(cards))}

    def prepare_state(self, curr_hand, table_cards, personal_history, opponent_history):
        curr_hand_map = [self.card_map[card] for card in curr_hand]
        table_cards_map = [self.card_map[card] for card in table_cards]
        table_cards_map += [-1] * (5 - len(table_cards_map))
        return curr_hand_map + table_cards_map  # + personal_history   + opponent_history

    def action(self, curr_hand, table_cards, personal_history, opponent_history, current_state, single_max_raise):
        # state = self.prepare_state(curr_hand, table_cards, personal_history, opponent_history)
        state = current_state.copy()
        if len(state) < 10:
            state += [0] * (10 - len(state))
        state.append(single_max_raise)
        self.state_memory.append(state)
        state_array = np.array(state).reshape(self.state_size)

        if np.random.uniform(0, 1) < self.anticipatory:
            if np.random.uniform(0, 1) > self.epsilon:
                action = self.br_model.predict(state_array)[0]
                # print('br action:', action)

                # if action.sum() == 0:
                #     action[np.random.randint(0, 3)] += 1
                sl_action = [0, 0, 0]
                sl_action[np.argmax(action)] += 1
                # self.sl_memory.append((state, sl_action))


                # print(action)

            else:
                rand_action = np.random.randint(0, 3)
                action = [0, 0, 0]
                action[rand_action] += 1

        else:
            action = self.ar_model.predict(state_array)[0]
        # print('action:', action)
        # sl_action = [0, 0, 0]
        # sl_action[np.argmax(action)] += 1
        return np.argmax(action), action

    def save_data(self, action_history, reward_history):
        # print(action_history, reward_history)
        target_q = self.q_br_model.predict(np.array(self.state_memory))
        for i in range(len(self.state_memory)):
            # potential to not work
            if i == len(self.state_memory) - 1:
                state_reward = reward_history[i]
            else:
                state_reward = reward_history[i] + self.bellman_value * self.q_br_model.predict(
                    np.array(self.state_memory[i + 1]).reshape(self.state_size)).max()
                # print('reward_diff:',reward_history[i], state_reward )
            current_action = action_history[i]
            # print(state_reward)
            # could mess up
            target_q[i][current_action] = state_reward
            # print(self.state_memory[i], target_q[i])
            self.rl_memory.append((self.state_memory[i], target_q[i]))

    def load_data(self, memory):
        if len(memory) < self.batch_size:
            idx = np.random.choice(range(len(memory)), len(memory), False)
        else:
            idx = np.random.choice(range(len(memory)), self.batch_size, False)
        train_x = [memory[i][0] for i in idx]
        train_y = [memory[i][1] for i in idx]
        return train_x, train_y

    def train_br(self):
        if len(self.rl_memory) > self.rl_buffer_size:
            self.rl_memory = self.rl_memory[-self.rl_buffer_size:]
        train_x, train_y = self.load_data(self.rl_memory)
        pd.DataFrame(np.concatenate((np.array(train_x), np.array(train_y)), axis=1)).to_csv('action_reward.csv')
        self.br_model.fit(np.array(train_x), np.array(train_y), epochs=2, verbose=2)
        self.epsilon *= 0.999

    def train_ar(self):
        if len(self.sl_memory) > self.sl_buffer_size:
            self.sl_memory = self.sl_memory[-self.sl_buffer_size:]
        train_x, train_y = self.load_data(self.sl_memory)
        self.ar_model.fit(np.array(train_x), np.array(train_y), epochs=2, verbose=2)


    def train_q_br(self):
        weights = self.br_model.get_weights()
        target_weights = self.q_br_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.q_br_model.set_weights(target_weights)

    def swap(self, dominant_player):
        #todo should do merge
        weights = dominant_player.br_model.get_weights()
        target_weights = self.br_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        weights = dominant_player.ar_model.get_weights()
        target_weights = self.ar_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.ar_model.set_weights(target_weights)
        self.rl_memory = dominant_player.rl_memory.copy()
        self.sl_memory = dominant_player.sl_memory.copy()
        self.train_q_br()


if __name__ == '__main__':
    NFSP()
