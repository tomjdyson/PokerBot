import numpy as np
from collections import Counter
from PokerBot.poker_player import PokerPlayer
from random import shuffle, randint


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

    @staticmethod
    def create_cards():
        # change for aces
        numbers = list(range(1, 14))
        suits = ['c', 'h', 's', 'd']
        cards = [(i, j) for i in numbers for j in suits]
        return cards

    def pick_cards(self, n):
        cards = [self.remaining_cards.pop(randint(0, len(self.remaining_cards) - 1)) for i in range(n)]
        return cards

    @staticmethod
    def count_pairs(state):
        c = Counter([i[0] for i in state])

        pairs_list = [i for i in c if c[i] >= 2]
        three_kind_list = [i for i in c if c[i] >= 3]
        four_kind_list = [i for i in c if c[i] >= 4]
        if len(pairs_list) >= 2:
            two_pair = 4

            if len(four_kind_list) > 0:
                four_kind = 4
                high_card = max(four_kind_list)
                hand_rank = 2

            elif len(three_kind_list) > 0:
                four_kind = 3
                high_card = max(three_kind_list)
                hand_rank = 6

            else:
                four_kind = 2
                high_card = max(pairs_list)
                hand_rank = 7

        elif len(pairs_list) == 1:
            two_pair = 3

            if len(four_kind_list) > 0:
                four_kind = 4
                high_card = max(four_kind_list)
                hand_rank = 2

            elif len(three_kind_list) > 0:
                four_kind = 3
                high_card = max(three_kind_list)
                hand_rank = 6

            else:
                four_kind = 2
                high_card = max(pairs_list)
                hand_rank = 8

        else:
            two_pair, four_kind = 1, 1,
            high_card = max([i for i in c])
            hand_rank = 9
        return [high_card], [two_pair, four_kind], hand_rank

    @staticmethod
    def count_flush(state):
        c = Counter([i[1] for i in state])
        high_card = 0
        hand_rank = 10
        if max(c.values()) >= 5:
            hand_rank = 4
            for i in c:
                if c[i] >= 5:
                    high_card = max([j[0] for j in state if j[1] == i])
                    break
        return [high_card], [min(max(c.values()), 5)], hand_rank

    @staticmethod
    def cards_in_row(state):
        straight_val = 1
        straight_max = 1
        high_card = 0
        for i in range(len(state) - 1):
            curr_value = state[i]
            j = i + 1
            next_value = state[j]
            if curr_value == next_value:
                continue
            elif next_value == curr_value + 1:
                straight_val += 1
                high_card = next_value
            else:
                straight_val = 1
            straight_max = max([straight_val, straight_max])
        return straight_max, high_card

    def count_straight(self, state):
        sorted_hand = [i[0] for i in state]
        sorted_hand.sort()
        c = Counter([i[1] for i in state])
        for i in c:
            if c[i] >= 5:
                straight_flush = [j[0] for j in state if j[1] == i]
                straight_flush.sort()
                straight_max, high_card = self.cards_in_row(straight_flush)
                if straight_max >= 5:
                    straight_max = 6
                    return [high_card], [straight_max], 1

        hand_rank = 10
        straight_max, high_card = self.cards_in_row(sorted_hand)
        if straight_max >= 5:
            hand_rank = 5
            straight_max = 5

        return [high_card], [straight_max], hand_rank

    def find_state(self, cards):
        pair_high, pairs, pair_rank = self.count_pairs(cards)
        straight_high, straight, straight_rank = self.count_straight(cards)
        flush_high, flush, flush_rank = self.count_flush(cards)
        best_hand = min(pair_rank, straight_rank, flush_rank)
        if best_hand == pair_rank:
            return pair_high + pairs + straight + flush, pair_rank
        elif best_hand == straight_rank:
            return straight_high + pairs + straight + flush, straight_rank
        else:
            return flush_high + pairs + straight + flush, flush_rank

    def find_best(self):
        try:
            top_rank = min([i.curr_rank for i in self.player_list])
        except Exception as e:
            # print([player.name for player in self.player_list])
            print(self.curr_max_bet, self.big_blind, self.small_blind)
            print({player.name : [player.previous_action, player.curr_bet] for player in self.starting_player_list})
            raise e
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
