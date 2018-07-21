import numpy as np
import pandas as pd
from PokerBot.agent_bet import PokerAgentBet
from collections import Counter
import time
from random import shuffle


class PokerAgentBetRL(PokerAgentBet):
    def initialize_agent(self):
        self.player_list = self.frozen_player_list.copy()
        shuffle(self.player_list)
        self.starting_player_list = self.player_list
        self.small_blind = 1
        self.big_blind = 2
        self.curr_cards = None
        self.remaining_cards = None
        self.all_cards = self.create_cards()
        self.table_cards = None
        self.curr_bets = None
        self.update_list = []
        self.curr_pot = self.big_blind + self.small_blind
        self.curr_max_bet = 0
        self.single_max_raise = 0

    def rl_reward(self, player, curr_min, curr_max):
        new_min_pos = player.curr_money - curr_min
        new_max_pos = curr_max - player.curr_money

        if new_min_pos >= player.min_pos:
            if len(player.game_hands) > 0:
                player.train()

        elif new_max_pos <= player.max_pos:
            if len(player.game_hands) > 0:
                player.train()

        player.min_pos = new_min_pos
        player.max_pos = new_max_pos

    def run_tournament_rl(self):
        hand_counter = 1
        while len(self.starting_player_list) > 1:
            # print(hand/_counter)
            self.run_game()
            curr_min = min([player.curr_money for player in self.starting_player_list])
            curr_max = max([player.curr_money for player in self.starting_player_list])
            for player in self.starting_player_list:
                # print(player.name, player.curr_money, player.curr_bet, self.curr_pot)
                self.rl_reward(player, curr_max=curr_max, curr_min=curr_min)
                if player.curr_money == 0:
                    self.starting_player_list.remove(player)
                if player.curr_money < 0:
                    ValueError('Cant be in debt')
            if hand_counter % 50 == 0:
                self.big_blind *= 2
                self.small_blind *= 2
            hand_counter += 1
        return self.starting_player_list

    def initialize_players(self):
        for player in self.player_list:
            player.curr_hand = None
            player.start_money = 1000
            player.curr_money = player.start_money
            player.curr_bet = 0
            player.curr_state = None
            player.curr_rank = None
            player.previous_action = None
            player.bet = 0
            player.update_dict = {'final_state': None, 'opening_state': None, 'flop_state': None,
                                  'turn_state': None}
            player.non_blind = 1
            player.non_blind_play = 0
            player.min_pos = 0
            player.max_pos = 0
            player.game_hands = []

    def initialize_rl_tournament(self):
        self.initialize_agent()
        self.initialize_players()
        # TODO Reset players to initial values - just copy init on base and player

    def simulate_tournaments_rl(self, n):
        self.frozen_player_list = self.starting_player_list
        print(self.frozen_player_list)
        winner_list = []
        for i in range(n):
            self.initialize_rl_tournament()
            winner = self.run_tournament_rl()[0]
            winner_list.append(winner.name)
            pd.DataFrame(winner.tournament_hands).to_csv('rl_winner.csv')
            print(winner.name)
            print(Counter(winner_list))


if __name__ == '__main__':
    rl_agent = PokerAgentBetRL(5)
    rl_agent.simulate_tournaments_rl(1000)

    # todo just loop and print last 10 win rates??
