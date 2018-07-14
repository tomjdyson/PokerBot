import numpy as np
from PokerBot.player_bet import SimpleBet, SimpleModelBet, SimpleNNBet, NNRLBet
import pandas as pd


class PokerPlayer:
    def __init__(self, name, bet_style='model', bet_obj = NNRLBet(10), action_clf=None, bet_clf=None):
        # self.name = np.random.choice(
        #     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'], 1)
        self.name = name
        self.curr_hand = None
        self.start_money = 1000
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

        self.tournament_hands = []
        self.game_hands = []

        if bet_style == 'simple':

            self.betting_obj = SimpleBet(np.random.randint(-100, -40) / 10, np.random.randint(-39, 20) / 10,
                                         np.random.randint(1, 1000) / 10)

        else:
            self.betting_obj = bet_obj
            # self.betting_obj = SimpleBet(-1000, -500,
            #                              4)
            # call_risk AND raise_risk

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

        return self.previous_action, self.bet

    def train(self):
        self.betting_obj.train(pd.DataFrame(self.game_hands))


if __name__ == '__main__':
    pkr = PokerPlayer('a', 'model')
    pkr.curr_state = [1, 3, 2, 2, 2, 1, 1, 1, 2, 2]
    print(pkr.table_risk('final_state'))

    # print(a)
