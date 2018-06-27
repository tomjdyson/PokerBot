import numpy as np
from PokerBot.player_bet import SimpleBet, SimpleModelBet
import pandas as pd


class PokerPlayer:
    def __init__(self, name, bet_style='simple', action_clf = None, bet_clf = None):
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

        # will have to have a file location somewhere
        self.stat_dict = {'final_state': pd.read_csv('final_state_df.csv'),
                          'opening_state': pd.read_csv('opening_state_df.csv'),
                          'flop_state': pd.read_csv('flop_state_df.csv'),
                          'turn_state': pd.read_csv('turn_state_df.csv')}

        self.tournament_hands = []

        if bet_style == 'simple':

            self.betting_obj = SimpleBet(np.random.randint(-100, -40) / 10, np.random.randint(-39, 20) / 10,
                                         np.random.randint(1, 1000) / 10)

        else:
            self.betting_obj = SimpleModelBet(action_clf = action_clf, bet_clf = bet_clf)
            # self.betting_obj = SimpleBet(-1000, -500,
            #                              4)
            # call_risk AND raise_risk

    def table_risk(self, game_state):
        state_df = self.stat_dict[game_state].values

        if game_state == 'opening_state':
            risk = 0

        stat_array = state_df[np.all(state_df[:, 4:-2] == np.array(self.curr_state)[4:], axis=1), -2:].sum(axis=0)
        return stat_array[1] / stat_array[0]

    def decide_action(self, game_state, big_blind, curr_max_bet, curr_table, remaining_player_hands,
                      remaining_players_tournament, hand_lowest_money, single_max_raise):
        state_df = self.stat_dict[game_state].values

        if self.curr_money < 0:
            print({'player': self.name,
                   'start_money': self.start_money,
                   'curr_money': self.curr_money,
                   'max_bet': curr_max_bet, 'remaining_players_hand': remaining_player_hands,
                   'curr_pot': curr_table,
                   'remaining_players_tournament': remaining_players_tournament,
                   'hand_lowest_money': hand_lowest_money, 'curr_bet': self.curr_bet,
                   'single_max_raise': single_max_raise})
            raise (ValueError, "Can't be in debt")

        try:
            stat_array = state_df[np.all(state_df[:, :-2] == np.array(self.curr_state), axis=1), -2:][0]
            risk = stat_array[1] / stat_array[0]

        except Exception as e:
            print(e)
            risk = 0.2

        self.previous_action, self.bet = self.betting_obj.action(self_risk=risk, big_blind=big_blind,
                                                                 max_bet=curr_max_bet,
                                                                 curr_pot=curr_table, curr_bet=self.curr_bet,
                                                                 table_risk=self.table_risk(game_state),
                                                                 curr_money=self.curr_money,
                                                                 remaining_players_hand=remaining_player_hands,
                                                                 remaining_players_tournament=remaining_players_tournament,
                                                                 hand_lowest_money=hand_lowest_money,
                                                                 single_max_raise=single_max_raise)

        hand_stat = {'player': self.name, 'self_risk': risk, 'table_risk': self.table_risk(game_state),
                     'curr_money': self.curr_money,
                     'max_bet': curr_max_bet, 'remaining_players_hand': remaining_player_hands,
                     'curr_pot': curr_table,
                     'remaining_players_tournament': remaining_players_tournament,
                     'hand_lowest_money': hand_lowest_money, 'curr_bet': self.curr_bet,
                     'bet': -1 if self.previous_action == 'fold' else self.bet,
                     'single_max_raise': single_max_raise}

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

        self.tournament_hands.append(hand_stat)

        return self.previous_action, self.bet


if __name__ == '__main__':
    pkr = PokerPlayer('a', 'model')
    pkr.curr_state = [1, 3, 2, 2, 2, 1, 1, 1, 2, 2]
    print(pkr.table_risk('final_state'))

    # print(a)
