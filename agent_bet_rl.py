import numpy as np
import pandas as pd
from PokerBot.agent_bet import PokerAgentBet
from collections import Counter
import time
from random import shuffle
import math


# TODO Initialize state as [-2]
# TODO CHange blind to player


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
        self.table_cards = []
        self.curr_bets = None
        self.update_list = []
        self.curr_pot = self.big_blind + self.small_blind
        self.curr_max_bet = 0
        self.single_max_raise = 0

    def rl_reward_position(self, player, curr_min, curr_max):
        new_min_pos = player.curr_money - curr_min
        new_max_pos = curr_max - player.curr_money
        new_pos = new_min_pos - new_max_pos
        player.curr_reward = new_pos - player.curr_pos
        # TODO Just record reward then train at the end of tournament - will be much faster
        if len(player.game_hands) > 0:
            reward_list = list(
                reversed([player.curr_reward * math.pow(0.7, i) for i in range(len(player.game_hands))]))
            game_hands_pd = pd.DataFrame(player.game_hands)
            game_hands_pd['reward'] = reward_list
            player.game_hands_list.append(game_hands_pd)
            # player.train()

        # if new_min_pos >= player.min_pos:
        #     if len(player.game_hands) > 0:
        #         player.train()
        #
        # elif new_max_pos <= player.max_pos:
        #     if len(player.game_hands) > 0:
        #         player.train()

        player.curr_pos_pos = new_pos

    def rl_reward_won_lost(self, player):
        player.curr_reward = player.start_money - player.curr_money
        if len(player.game_hands) > 0:
            reward_list = list(
                reversed([player.curr_reward * math.pow(0.7, i) for i in range(len(player.game_hands))]))
            game_hands_pd = pd.DataFrame(player.game_hands)
            game_hands_pd['reward'] = reward_list
            player.game_hands_list.append(game_hands_pd)

    def run_tournament_rl(self):
        hand_counter = 1
        hand_actions = []
        while len(self.starting_player_list) > 1:
            # print(hand/_counter)
            result, winner = self.run_game_rl()
            for player in winner:
                player.win_reward_history.append(self.curr_pot - player.pot_minus_1)
                if len(player.betting_obj.state_memory) > 0:
                    player.betting_obj.save_data(player.simple_action_history, player.win_reward_history)
            for player in self.starting_player_list:
                if len(player.betting_obj.state_memory) > 0:
                    if player not in winner:
                        player.betting_obj.save_data(player.simple_action_history, player.lose_reward_history)
                # print(player.name, player.curr_money, player.curr_bet, self.curr_pot)
                # self.rl_reward_won_lost(player)
                action_len = len(player.action_list)
                # print(action_len, len(player.action_value_list))
                hand_actions.append(pd.DataFrame(
                    {'hand_counter': [hand_counter] * action_len, 'hand': [player.curr_hand] * action_len,
                     'action': player.action_list, 'bet': player.bet_list, 'name': [player.name] * action_len,
                     'table_card': [self.table_cards] * action_len, 'state': player.state_list,
                     'action_values': player.action_value_list[0:action_len]
                     # 'win_reward': player.win_reward_history, 'lose_reward': player.lose_reward_history
                     }))
                player.bet_list = []
                player.action_list = []
                player.state_list = []
                if player.curr_money == 0:
                    self.starting_player_list.remove(player)
                if player.curr_money < 0:
                    ValueError('Cant be in debt')
                if self.curr_pot > 6000:
                    ValueError('Pot too large')

            if hand_counter % 50 == 0:
                self.big_blind *= 2
                self.small_blind *= 2
            hand_counter += 1
        return self.starting_player_list, hand_actions

    def initialize_players(self):
        rand_start = np.random.choice([True, False], 1)[0]
        for player in self.player_list:
            player.curr_hand = None
            # TODO: Not sure about this
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
            player.tournament_hands = []
            player.game_hands_list = []
            player.action_history = []
            player.simple_action_history = []
            player.reward_history = []
            # if rand_start:
            #     player.start_money = np.random.randint(100, 10000)
            #     player.curr_money = player.start_money

    def initialize_rl_tournament(self):
        self.initialize_agent()
        self.initialize_players()
        # TODO Reset players to initial values - just copy init on base and player

    def player_actions_rl(self, first):

        remove_list = []

        for player in self.player_list:
            opponent_list = self.player_list.copy()
            opponent_list.remove(player)

            opponent_history = [other_player.personal_history for other_player in opponent_list]
            opponent_history = [item for sublist in opponent_history for item in sublist]
            opponent_history += [-2] * (40 - len(opponent_history))

            action, bet = player.decide_action(self.table_cards, opponent_history, self.curr_max_bet,
                                               self.single_max_raise, self.curr_pot,
                                               len_player_list=len(self.player_list))

            # print('name', player.name, 'curr_money', player.curr_money, 'start_money', player.start_money, 'action',
            #       action, 'bet', bet, 'curr_bet',
            #       player.curr_bet, 'max_bet', self.curr_max_bet, 'curr_pot', self.curr_pot)
            if action == 'fold':
                remove_list.append(player)
                if len(self.player_list) - len(remove_list) == 1:
                    break
                continue

            self.curr_pot += bet
            self.single_max_raise = max(self.single_max_raise, bet - self.single_max_raise)
            self.curr_max_bet = max(self.curr_max_bet, player.curr_bet)
            if not first:
                if all([((player.curr_bet == self.curr_max_bet) | (player.curr_bet == player.start_money)) for player in
                        self.player_list]):
                    break
        [self.player_list.remove(player) for player in remove_list]

    def run_betting_rl(self):

        if len(self.player_list) > 1:
            equal = False
            first = True
            while not equal:
                self.player_actions_rl(first)
                equal = self.all_players_bet()
                first = False

    def initialize_game_rl(self):

        for player in self.starting_player_list:
            player.start_money = player.curr_money
            player.curr_bet = 0
            player.game_hands = []
            player.personal_history = [-2 for i in range(player.history_length)]
            player.action_history = []
            player.simple_action_history = []
            player.reward_history = []
            player.betting_obj.state_memory = []
            player.previous_action = None
            player.win_reward_history = []
            player.lose_reward_history = []
            player.pot_minus_1 = 0
            player.pot_minus_2 = 0

        self.player_list = self.starting_player_list
        self.curr_pot = 0
        self.curr_pot = self.big_blind + self.small_blind
        self.single_max_raise = self.big_blind
        self.table_cards = []
        if self.dealer >= len(self.starting_player_list) - 1:
            self.dealer = 0
        else:
            self.dealer += 1
        if len(self.player_list) > 2:
            big_blind_pos = self.dealer - (len(self.player_list) - 2) if self.dealer >= (
                len(self.player_list) - 2) else self.dealer + 2
            small_blind_pos = self.dealer - (len(self.player_list) - 1) if self.dealer >= (
                len(self.player_list) - 1) else self.dealer + 1
            self.player_list = self.starting_player_list[big_blind_pos + 1:] + self.starting_player_list[
                                                                               :big_blind_pos + 1]

        else:
            small_blind_pos = self.dealer
            big_blind_pos = (self.dealer + 1) % 2
            self.player_list = [self.starting_player_list[small_blind_pos]] + [self.starting_player_list[big_blind_pos]]
        self.starting_player_list[big_blind_pos].add_blind(self.big_blind)
        self.starting_player_list[small_blind_pos].add_blind(self.small_blind)
        self.curr_max_bet = max(player.curr_bet for player in self.starting_player_list)

    def run_game_rl(self):
        self.initialize_game_rl()
        self.remaining_cards = self.create_cards()
        for i in self.player_list:
            hand = self.pick_cards(2)
            i.curr_hand = hand
        self.fill_in_state('opening_state')
        # while loop if all players not on same amount - stick in betting so not infinite loop
        # insert bet
        self.run_betting_rl()
        self.table_cards += self.pick_cards(3)
        self.fill_in_state('flop_state')
        # insert bet
        self.run_betting_rl()
        self.table_cards += self.pick_cards(1)
        self.fill_in_state('turn_state')
        # insert bet
        self.run_betting_rl()
        self.table_cards += self.pick_cards(1)
        # insert_bet
        self.fill_in_state('final_state')
        self.run_betting_rl()
        result, player_list = self.find_best()
        self.handle_money(result, player_list)
        return result, player_list

    def simulate_tournaments_rl(self, n):
        self.frozen_player_list = self.starting_player_list
        winner_list = []
        for i in range(n):
            self.initialize_rl_tournament()
            winner, hand_list = self.run_tournament_rl()
            winner = winner[0]
            # winner.reward_list = winner.reward_history * -1
            hand_history_df = pd.concat(hand_list, axis=0)
            hand_history_df.to_csv('hand_history.csv')
            winner_list.append(winner.name)
            if (len(winner_list) % 50 == 0) & (len(winner_list) > 1):
                last_50 = winner_list[-50:]
                last_50_count = Counter(last_50)
                dominant = [j for j in last_50_count.keys() if last_50_count[j] == max(last_50_count.values())][0]
                for player in self.frozen_player_list:
                    if player.name not in last_50:
                        print(player.name, ' swapped with ', dominant)
                        player.swap(
                            [dom_player for dom_player in self.frozen_player_list if dom_player.name == dominant][0])
                        break

            # # for player in self.frozen_player_list:
            # #     player.batch_train()
            # for player in self.frozen_player_list:
            #     if len(player.betting_obj.state_memory) > 0:
            #         player.betting_obj.save_data(player.simple_action_history, player.reward_history)
            # pd.DataFrame(winner.tournament_hands).to_csv('rl_winner_2.csv')
            # if (i % 2 == 0) & (i > 1):
            for player in self.frozen_player_list:
                player.betting_obj.train_br()
                # player.betting_obj.train_ar()
            if (i % 50 == 0) & (i > 1):
                print('retraining q_br')
                for player in self.frozen_player_list:
                    player.betting_obj.train_q_br()
            print(winner.name)
            print(Counter(winner_list))


if __name__ == '__main__':
    rl_agent = PokerAgentBetRL(5)
    rl_agent.simulate_tournaments_rl(10000)

    # todo just loop and print last 10 win rates??
