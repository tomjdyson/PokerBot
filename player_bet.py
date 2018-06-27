import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
import pandas as pd


class SimpleBet:
    def __init__(self, call_cost, raise_cost, opponent_risk):
        self.call_cost = call_cost
        self.raise_cost = raise_cost
        self.opponent_risk = opponent_risk

    def action(self, self_risk, table_risk, curr_money, max_bet, remaining_players_hand, curr_pot,
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise):
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

        if bet < -1:
            print(bet, max_bet, curr_bet)
            raise (ValueError, 'Cant bet negative')

        return action, bet



# action_clf = RandomForestClassifier()
# bet_clf = RandomForestRegressor()
# train = pd.read_csv('C:/Users/dysont/Documents/Graduate/rl/PokerBot/tournament_data.csv').sample(1000000)
# train['action'] = 0
# train.loc[train.bet == 0, 'action'] = 1
# train.loc[train.bet == train.max_bet - train.curr_bet, 'action'] = 1
# train.loc[train.bet + train.curr_bet > train.max_bet, 'action'] = 2
# bet_train = train[train.action == 2]
#
#
# X = train.drop(['Unnamed: 0', 'bet', 'player', 'action', 'single_max_raise'], axis=1)
# y = train['action']
# bet_X = bet_train.drop(['Unnamed: 0', 'bet', 'player', 'action', 'single_max_raise'], axis=1)
# bet_y = bet_train['bet']
# action_clf.fit(X, y)
# bet_clf.fit(bet_X, bet_y)


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
               remaining_players_tournament, hand_lowest_money, curr_bet, big_blind, single_max_raise):

        predict_array = pd.DataFrame({'curr_bet': curr_bet, 'curr_money': curr_money, 'curr_pot': curr_pot,
                                      'hand_lowest_money': hand_lowest_money, 'max_bet': max_bet,
                                      'remaining_players_hand': remaining_players_hand,
                                      'remaining_players_tournament': remaining_players_tournament,
                                      'self_risk': self_risk,
                                      'table_risk': table_risk}, index=[0])




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


if __name__ == '__main__':
    pkr = SimpleModelBet()
