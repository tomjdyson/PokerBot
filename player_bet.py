import pandas as pd
import numpy as np


class SimpleBet:
    def __init__(self, call_cost, raise_cost, opponent_risk):
        self.call_cost = call_cost
        self.raise_cost = raise_cost
        self.opponent_risk = opponent_risk

    def decide_bet(self, risk, curr_table_bet):

        return

    # def call_cost(self, risk,):

    def action(self, risk, big_blind, curr_max_bet, curr_table, curr_self_bet):
        # must bet same amount as big blind
        win_call_cost = (curr_table - curr_self_bet) * risk
        loss_call_cost = (curr_max_bet) * (1 - risk)
        call_action = 'call'
        fold_action = 'fold'
        # print(win_call_cost, loss_call_cost)

        if curr_max_bet == curr_self_bet:
            call_action = 'check'
            fold_action = 'check'

        if win_call_cost - loss_call_cost > self.call_cost:
            if win_call_cost > loss_call_cost + self.raise_cost:
                bet = curr_max_bet + (big_blind * round((self.opponent_risk * risk) - (1 - risk)))
                action = 'raise'
            else:
                bet = curr_max_bet - curr_self_bet
                action = call_action
        else:
            bet = 0
            action = fold_action
        return action, bet
