import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class RFModel:
    def __init__(self, filepath, sample=None):
        self.train_data = pd.read_csv(filepath)
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
