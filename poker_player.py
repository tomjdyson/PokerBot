import numpy as np
from PokerBot.player_bet import SimpleBet, SimpleModelBet, SimpleNNBet, NNRLBet, RandomBet
import pandas as pd
import random


class PokerPlayer:
    def __init__(self, name, bet_style='model', bet_obj=None, action_clf=None, bet_clf=None, keep_rate=0.3):
        self.name = name
        self.curr_hand = None
        self.start_money = 1000
        self.keep_rate = keep_rate
        self.curr_money = self.start_money
        self.curr_bet = 0
        self.curr_state = None
        self.curr_rank = None
        self.previous_action = None
        self.bet = 0
        self.update_dict = {'final_state': None, 'opening_state': None, 'flop_state': None,
                            'turn_state': None}
        self.non_blind = 1
        self.non_blind_play = 0

        self.min_pos = 0
        self.max_pos = 0
        self.curr_pos = 0
        self.curr_reward = 0
        self.reward_list = []
        self.random_game = False

        self.tournament_hands = []
        self.game_hands = []
        self.game_hands_list = []
        self.action_list = []
        self.bet_list = []
        self.state_list = []

        if bet_style == 'simple':

            self.betting_obj = SimpleBet(np.random.randint(-100, -40) / 10, np.random.randint(-39, 20) / 10,
                                         np.random.randint(1, 1000) / 10)

        else:
            self.betting_obj = bet_obj

    def table_risk(self, game_state, stat_dict):
        state_df = stat_dict[game_state]

        if game_state == 'opening_state':
            return 0

        stat_array = state_df[np.all(state_df[:, 4:-2] == self.curr_state[4:], axis=1), -2:].sum(axis=0)
        # stat_array = state_df[state_df[:, 4-2] == self.curr_state[4:], -2:].toarray().sum(axis=0)
        return stat_array[1] / stat_array[0]

    def decide_action(self, game_state, big_blind, curr_max_bet, curr_table, remaining_player_hands,
                      remaining_players_tournament, hand_lowest_money, single_max_raise, vips_list, stat_dict):
        state_df = stat_dict[game_state]

        try:
            stat_array = state_df[np.all(state_df[:, :-2] == self.curr_state, axis=1), -2:][0]
            risk = stat_array[1] / stat_array[0]

        except Exception as e:
            print(e)
            print(self.curr_hand)
            print(self.curr_state)
            risk = 0.2

        new_vips_list = [0, 0, 0, 0, 0]
        for i in range(len(vips_list)):
            new_vips_list[i] = vips_list[i]

        if not self.random_game:

            self.previous_action, self.bet = self.betting_obj.action(self_risk=risk, big_blind=big_blind,
                                                                     max_bet=curr_max_bet,
                                                                     curr_pot=curr_table, curr_bet=self.curr_bet,
                                                                     table_risk=self.table_risk(game_state, stat_dict),
                                                                     curr_money=self.curr_money,
                                                                     remaining_players_hand=remaining_player_hands,
                                                                     remaining_players_tournament=remaining_players_tournament,
                                                                     hand_lowest_money=hand_lowest_money,
                                                                     single_max_raise=single_max_raise,
                                                                     new_vips_list=new_vips_list)
        else:
            self.previous_action, self.bet = RandomBet.action(self_risk=risk, big_blind=big_blind,
                                                              max_bet=curr_max_bet,
                                                              curr_pot=curr_table, curr_bet=self.curr_bet,
                                                              table_risk=self.table_risk(game_state, stat_dict),
                                                              curr_money=self.curr_money,
                                                              remaining_players_hand=remaining_player_hands,
                                                              remaining_players_tournament=remaining_players_tournament,
                                                              hand_lowest_money=hand_lowest_money,
                                                              single_max_raise=single_max_raise,
                                                              new_vips_list=new_vips_list)

        if self.curr_bet == 0:
            self.non_blind += 1
            if self.bet > 0:
                self.non_blind_play += 1

        hand_stat = {'player': self.name, 'self_risk': risk, 'table_risk': self.table_risk(game_state, stat_dict),
                     'curr_money': self.curr_money,
                     'max_bet': curr_max_bet, 'remaining_players_hand': remaining_player_hands,
                     'curr_pot': curr_table,
                     'remaining_players_tournament': remaining_players_tournament,
                     'hand_lowest_money': hand_lowest_money, 'curr_bet': self.curr_bet,
                     'bet': -1 if self.previous_action == 'fold' else self.bet,
                     'single_max_raise': single_max_raise,
                     'vips_1': new_vips_list[0],
                     'vips_2': new_vips_list[1],
                     'vips_3': new_vips_list[2],
                     'vips_4': new_vips_list[3],
                     'vips_5': new_vips_list[4],
                     }

        if self.bet > self.curr_money:
            # print('all in')
            self.bet = self.curr_money
            hand_stat['bet'] = self.bet
            self.curr_bet = self.start_money
            self.previous_action = 'raise'
            self.curr_money = 0

        else:
            self.curr_bet += self.bet
            self.curr_money -= self.bet
        # print(hand_stat)
        self.tournament_hands.append(hand_stat)
        self.game_hands.append(hand_stat)
        self.action_list.append(self.previous_action)
        self.bet_list.append(-1 if self.previous_action == 'fold' else self.bet)
        self.state_list.append(game_state)
        return self.previous_action, self.bet

    def train(self):
        self.betting_obj.train(pd.DataFrame(self.game_hands), self.curr_reward)

    def batch_train(self):
        train_hands = pd.concat(self.game_hands_list, axis=0)
        train_hands = train_hands.sample(frac=self.keep_rate)
        self.betting_obj.train(train_hands.drop('reward', axis=1), train_hands.reward)


class QPokerPlayer:
    def __init__(self, name, history_length, betting_obj):
        self.name = name
        self.curr_hand = None
        self.start_money = 1000
        self.curr_money = self.start_money
        self.curr_bet = 0
        self.curr_rank = None
        self.curr_state = []
        self.previous_action = None
        self.bet = 0
        # todo make these into one tuple
        self.action_history = []
        self.reward_history = []
        self.lose_reward_history = []
        self.win_reward_history = []
        self.simple_action_history = []
        self.action_list = []
        self.bet_list = []
        self.state_list = []
        self.state_record = {'leave': [], 'play': []}
        self.epsilon = 0.1
        self.history_length = history_length
        self.personal_history = [-2 for i in range(history_length)]
        self.betting_obj = betting_obj
        self.pot_minus_1 = 0
        self.pot_minus_2 = 0
        self.action_value_list = []

    def add_blind(self, blind_value):
        self.curr_bet = min(blind_value, self.start_money)
        self.curr_money -= min(blind_value, self.start_money)
        self.reward_history.append(self.curr_bet)
        self.lose_reward_history.append(-self.curr_bet)
        self.action_history.append(self.curr_bet)
        # TODO Need to see if this is correct - is only necessary when folding
        # self.simple_action_history.append(1)
        self.update_personal()

    def update_personal(self):
        self.personal_history = self.action_history.copy()
        self.personal_history = self.personal_history[:self.history_length]
        self.personal_history += [-2] * (self.history_length - len(self.personal_history))

    def decide_action(self, table_cards, opponent_history, current_max_bet, single_max_raise, current_pot,
                      len_player_list):
        # print(self.curr_state)
        if self.previous_action is not None:
            if self.previous_action == 'raise':
                # self.win_reward_history.append(self.pot_minus_1 - self.pot_minus_2)
                self.win_reward_history.append(
                    (self.pot_minus_1 - self.pot_minus_2) + (self.prev_raise * len_player_list))
            if self.previous_action == 'call':
                self.win_reward_history.append(self.pot_minus_1 - self.pot_minus_2)

        # todo need to decay epsilon
        # if random.uniform(0, 1) > self.epsilon:
        # print(self.curr_hand, table_cards)
        action, action_values = self.betting_obj.action(self.curr_hand, table_cards,
                                                        self.personal_history, opponent_history, self.curr_state,
                                                        self.curr_bet)
        # else:
        #     # TODO Move random into here
        #     action = np.random.randint(0, 3)
        self.simple_action_history.append(action)
        if action == 0:
            self.previous_action = 'fold'
            self.bet = 0
            self.state_record['leave'].append(self.curr_state[0:5])
            if current_max_bet == self.curr_bet:
                self.previous_action = 'call'
        elif action == 1:
            self.previous_action = 'call'
            self.bet = current_max_bet - self.curr_bet
            self.state_record['play'].append(self.curr_state[0:5])

        elif action == 2:
            self.previous_action = 'raise'
            # self.bet = (current_max_bet - self.curr_bet) + np.random.randint(1, 3) * single_max_raise
            self.bet = (current_max_bet - self.curr_bet) + single_max_raise
            self.state_record['play'].append(self.curr_state[0:5])


        if self.bet > self.curr_money:

            self.bet = self.curr_money
            self.curr_bet = self.start_money
            self.previous_action = 'raise'
            self.curr_money = 0

        else:
            self.curr_bet += self.bet
            self.curr_money -= self.bet

        if self.bet > 100000:
            self.bet = self.curr_money
            self.curr_money = 0
            raise ValueError('Too large values')

        # print(self.previous_action)

        self.prev_raise = self.bet - current_max_bet
        self.action_history.append(-1 if self.previous_action == 'fold' else self.bet)
        self.update_personal()
        # if self.previous_action != 'fold':
        self.reward_history.append(self.bet)
        # THis doesnt work perfectly but works for now
        self.lose_reward_history.append(-self.bet)
        self.action_list.append(self.previous_action)
        self.bet_list.append(-1 if self.previous_action == 'fold' else self.bet)
        self.state_list.append(self.curr_hand + table_cards)
        self.action_value_list.append(action_values)
        self.pot_minus_1 = current_pot
        self.pot_minus_2 = self.pot_minus_1
        # print(self.name, self.previous_action)

        return self.previous_action, self.bet

    def train(self):
        self.betting_obj.train(self.reward_history * -1)

    def swap(self, dominant_player):
        self.betting_obj.swap(dominant_player.betting_obj)


if __name__ == '__main__':
    pkr = PokerPlayer('a', 'model')
    pkr.curr_state = [1, 3, 2, 2, 2, 1, 1, 1, 2, 2]
    print(pkr.table_risk('final_state'))

    # print(a)
