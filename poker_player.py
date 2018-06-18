import numpy as np
from PokerBot.player_bet import SimpleBet
import pandas as pd


class PokerPlayer:
    def __init__(self, betting_obj=SimpleBet(0.3, 0.5, 5)):
        self.name = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], 1)
        self.curr_hand = None
        self.start_money = 1000
        self.curr_money = self.start_money
        self.curr_bet = 0
        self.curr_state = None
        self.curr_rank = None
        self.previous_action = None
        self.update_dict = {'final_state': None, 'opening_state': None, 'flop_state': None,
                            'turn_state': None}

        # will have to have a file location somewhere
        self.stat_dict = {'final_state': pd.read_csv('final_state_df.csv'),
                          'opening_state': pd.read_csv('opening_state_df.csv'),
                          'flop_state': pd.read_csv('flop_state_df.csv'),
                          'turn_state': pd.read_csv('turn_state_df.csv')}

        self.betting_obj = SimpleBet(np.random.randint(-80, -30) / 10, np.random.randint(-29, 20) / 10,
                                     np.random.randint(1, 40) / 10)
        # self.betting_obj = SimpleBet(-1000, -500,
        #                              4)
        # call_risk AND raise_risk

    def table_risk(self, game_state):
        state_df = self.stat_dict[game_state].values

        if game_state == 'opening_state':
            risk = 0

        stat_array = state_df[np.all(state_df[:, 4:-2] == np.array(self.curr_state)[4:], axis=1), -2:].sum(axis = 0)
        return stat_array[1]/stat_array[0]


    def decide_action(self, game_state, big_blind, curr_max_bet, curr_table):
        state_df = self.stat_dict[game_state].values

        if self.start_money < 0:
            raise(ValueError, "Can't be in debt")

        try:
            stat_array = state_df[np.all(state_df[:, :-2] == np.array(self.curr_state), axis=1), -2:][0]
            risk = stat_array[1] / stat_array[0]

        except Exception as e:
            print(e)
            risk = 0.5

        self.previous_action, bet = self.betting_obj.action(risk=risk, big_blind=big_blind, curr_max_bet=curr_max_bet,
                                                            curr_table=curr_table, curr_self_bet=self.curr_bet)


        if bet >= self.curr_money:
            print('all in')
            self.curr_bet = self.start_money
            self.previous_action = 'raise'
            bet = self.curr_money

        self.curr_bet += bet
        self.curr_money -= bet

        return self.previous_action, bet

if __name__ == '__main__':
    pkr = PokerPlayer()
    pkr.curr_state = [1,3,2,2,2,1,1,1,2,2]
    print(pkr.table_risk('final_state'))

    # print(a)