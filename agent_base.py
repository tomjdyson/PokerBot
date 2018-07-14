import numpy as np
from collections import Counter
from PokerBot.poker_player import PokerPlayer
from random import shuffle, randint
import pandas as pd
import time
from scipy import sparse


class PokerAgentBase:
    def __init__(self, num_players, action_clf=None, bet_clf=None):
        self.player_1 = PokerPlayer('a')
        self.player_2 = PokerPlayer('b')
        self.player_3 = PokerPlayer('c')
        self.player_4 = PokerPlayer('d')
        self.player_5 = PokerPlayer('f')
        # self.player_5 = PokerPlayer('f', bet_style='model', action_clf=action_clf, bet_clf=bet_clf)
        # todo setter
        self.player_list = [self.player_1, self.player_2, self.player_3, self.player_4, self.player_5][0:num_players]
        shuffle(self.player_list)
        self.starting_player_list = self.player_list
        self.dealer = np.random.randint(0, num_players)
        self.small_blind = 1
        self.big_blind = 2
        self.curr_cards = None
        self.remaining_cards = None
        self.all_cards = self.create_cards()
        self.table_cards = None
        self.curr_bets = None
        self.update_list = []
        self.curr_pot = self.big_blind + self.small_blind
        self.stat_dict = {'opening_state': [], 'flop_state': [], 'turn_state': [], 'final_state': []}
        self.curr_max_bet = 0
        self.single_max_raise = 0
        self.game_state = None
        self.pair_dict = {0: 9, 1: 8, 2: 7, 3: 7}
        self.kind_dict = {1: 9, 2: 9, 3: 6, 4: 2}
        self.times = {'player_action': 0, 'initialize': 0, 'side_pool': 0, 'handle_money': 0, 'all_players_bet': 0,
                      'decide_action': 0, 'full_time': 0}
        self.stat_dict = {'final_state': pd.read_csv('final_state_df.csv').values,
                          'opening_state': pd.read_csv('opening_state_df.csv').values,
                          'flop_state': pd.read_csv('flop_state_df.csv').values,
                          'turn_state': pd.read_csv('turn_state_df.csv').values}
        self.counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}

    @staticmethod
    def create_cards():
        numbers = list(range(1, 14))
        suits = ['c', 'h', 's', 'd']
        cards = [(i, j) for i in numbers for j in suits]
        return cards

    def pick_cards(self, n):
        cards = [self.remaining_cards.pop(randint(0, len(self.remaining_cards) - 1)) for i in range(n)]
        return cards

    # @staticmethod
    # def count_pairs(state, pair_dict, kind_dict):
    #     c = Counter([i[0] for i in state])
    #     two_pair = len([i for i in c if c[i] >= 2])
    #     four_kind = max([c[i] for i in c])
    #     pair_rank = pair_dict[two_pair]
    #     kind_rank = kind_dict[four_kind]
    #     if (pair_rank == 7) & (kind_rank == 6):
    #         hand_rank = 4
    #         high_card = max([i for i in c if c[i] > 2])
    #     elif pair_rank == 9:
    #         hand_rank = kind_rank
    #         high_card = max([i for i in c])
    #     elif pair_rank < kind_rank:
    #         hand_rank = pair_rank
    #         high_card = max([i for i in c if c[i] == 2])
    #     else:
    #         hand_rank = kind_rank
    #         high_card = max([i for i in c if c[i] > 2])
    #
    #     return [high_card], [two_pair, four_kind], hand_rank

    def count_pairs_new(self, state):
        counts = self.counts.copy()
        two_pair, four_kind, pair_high, kind_high, card_high = 0, 1, 0, 0, 0

        for i in state:
            counts[i[0]] += 1
            if counts[i[0]] == 2:
                two_pair += 1
                pair_high, four_kind = max(pair_high, i[0]), max(four_kind, 2)
            elif counts[i[0]] > 2:
                four_kind, kind_high = max(four_kind, counts[i[0]]), max(kind_high, i[0])

        pair_rank, kind_rank = self.pair_dict[two_pair], self.kind_dict[four_kind]
        if (pair_rank == 7) & (kind_rank == 6):
            hand_rank, high_card = 4, kind_high
        elif pair_rank == 9:
            hand_rank, high_card = kind_rank, max([i[0] for i in state])
        elif pair_rank < kind_rank:
            hand_rank, high_card = pair_rank, pair_high
        else:
            hand_rank, high_card = kind_rank, kind_high

        return [high_card], [two_pair, four_kind], hand_rank

    def count_flush(self, state):
        high_card = 0
        hand_rank = 10
        state_count = {'c': 0, 'h': 0, 'd': 0, 's': 0}
        flush_state = 'c'
        for i in state:
            state_count[i[1]] += 1
            if state_count[i[1]] == 5:
                high_card = max([j[0] for j in state if j[1] == i[1]])
                hand_rank = 4
                flush_state = i

        return [high_card], [min(max(state_count.values()), 5)], hand_rank, flush_state

    # @staticmethod
    def cards_in_row(self, cards, suit=False):
        if suit:
            state = [j[0] for j in cards if j[1] == suit]
        else:
            state = [j[0] for j in cards]
        state.sort()

        straight_val = 1
        straight_max = 1
        high_card = 0
        hand_rank = 10
        for i in range(len(state) - 1):
            curr_value = state[i]
            j = i + 1
            next_value = state[j]
            if curr_value == next_value:
                continue
            elif next_value == curr_value + 1:
                straight_val += 1
            else:
                straight_val = 1
            straight_max = max([straight_val, straight_max])
            if straight_max == 5:
                high_card = next_value
                hand_rank = 5
                break

        return [high_card], [straight_max], hand_rank

    def find_state(self, cards):

        pair_high, pairs, pair_rank = self.count_pairs_new(cards)
        flush_high, flush, flush_rank, flush_suit = self.count_flush(cards)
        # straight_high, straight, straight_rank = self.cards_in_row(cards)
        if flush[0] == 5:
            straight_high, straight, straight_rank = self.cards_in_row(cards, flush_suit)
            if straight[0] < 5:
                straight_high, straight, straight_rank = self.cards_in_row(cards)
            else:
                straight_rank = 1
        else:
            straight_high, straight, straight_rank = self.cards_in_row(cards)

        best_hand = min(pair_rank, straight_rank, flush_rank)

        if best_hand == pair_rank:
            return pair_high + pairs + straight + flush, pair_rank
        elif best_hand == straight_rank:
            return straight_high + pairs + straight + flush, straight_rank
        else:
            return flush_high + pairs + straight + flush, flush_rank

    def find_best(self):

        top_rank = min([i.curr_rank for i in self.player_list])
        if Counter([i.curr_rank for i in self.player_list])[top_rank] > 1:
            drawn_rank = [i for i in self.player_list if i.curr_rank == top_rank]
            highest_card = max([i.curr_state[0] for i in drawn_rank])
            best_state = [i for i in drawn_rank if i.curr_state[0] == highest_card]
            if len(best_state) > 1:
                return 'draw', best_state
            else:
                return 'win', best_state
        else:
            return 'win', [i for i in self.player_list if i.curr_rank == top_rank]

    def fill_in_state(self, game_round):

        self.game_state = game_round
        for i in self.player_list:
            if game_round == 'opening_state':
                full_state, full_rank = self.find_state(i.curr_hand)
                i.update_dict[game_round] = full_state
                i.curr_state = full_state
                continue
            full_state, full_rank = self.find_state(i.curr_hand + self.table_cards)
            table_state, table_rank = self.find_state(self.table_cards)
            table_state[0] = 2 if table_state[0] == full_state[0] else 1
            i.curr_state = full_state + table_state
            i.curr_rank = full_rank
            i.update_dict[game_round] = full_state + table_state

    def run_game(self):
        self.remaining_cards = self.create_cards()
        for i in self.player_list:
            hand = self.pick_cards(2)
            i.curr_hand = hand
        self.fill_in_state('opening_state')
        # insert bet
        self.table_cards = self.pick_cards(3)
        self.fill_in_state('flop_state')
        # insert bet
        self.table_cards += self.pick_cards(1)
        self.fill_in_state('turn_state')
        # insert bet
        self.table_cards += self.pick_cards(1)
        # insert_bet
        self.fill_in_state('final_state')
        result, player_list = self.find_best()
        return result, player_list


if __name__ == '__main__':
    pkr = PokerAgentBase(5)
    print(pkr.times)

    # apair_dict = {0: 9, 1: 8, 2: 7}
    # akind_dict = {1: 9, 2: 9, 3: 6, 4: 2}
    # acounts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    #
    # # start_time = time.time()
    # # for i in range(1000000):
    # print(pkr.count_pairs_new([(13, 'c'), (13, 'c'), (13, 'd'), (13, 'd'), (9, 'c')]))
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print((time.time() - start_time) / 10000)

    # start_time = time.time()
    # for i in range(1000000):
    #     pkr.count_pairs([(13, 'c'), (1, 'c'), (8, 'd'), (11, 'd'), (2, 'c')], apair_dict, akind_dict)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print((time.time() - start_time) / 10000)
    # start_time = time.time()
    # for i in range(1000000):
    #     pkr.count_pairs_old([(13, 'c'), (1, 'c'), (8, 'd'), (11, 'd'), (2, 'c')])
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print((time.time() - start_time) / 10000)
    # print(pkr.count_pairs([(13, 'c'), (11, 'h'), (10, 'd'), (13, 'd'), (13, 'c')]))
