import numpy as np
from collections import Counter
import pickle
import pandas as pd


class PokerPlayer:
    def __init__(self):
        self.curr_hand = None
        self.start_money = 1000
        self.curr_money = self.start_money
        self.curr_bet = None
        self.curr_state = None
        self.curr_rank = None
        self.update_dict = {'final_state': None, 'opening_state': None, 'flop_state': None,
                            'turn_state': None}

    def reset_state(self):
        self.curr_hand = None
        self.curr_bet = None
        self.curr_state = None
        self.curr_rank = None


class PokerAgent:
    def __init__(self, num_players):
        self.player_1 = PokerPlayer()
        self.player_2 = PokerPlayer()
        self.player_3 = PokerPlayer()
        self.player_4 = PokerPlayer()
        self.player_5 = PokerPlayer()
        self.player_list = [self.player_1, self.player_2, self.player_3, self.player_4, self.player_5][0:num_players]
        self.small_blind = 1
        self.big_blind = 2
        self.curr_cards = None
        self.remaining_cards = None
        self.all_cards = self.create_cards()
        self.table_cards = None
        self.curr_bets = None
        self.update_list = []
        self.stat_dict = {'opening_state': [], 'flop_state': [], 'turn_state': [], 'final_state': []}

    def create_cards(self):
        # change for aces
        numbers = list(range(1, 14))
        suits = ['c', 'h', 's', 'd']
        cards = [(i, j) for i in numbers for j in suits]
        return cards

    def pick_cards(self, n):
        # fastest can be currently made
        cards = [self.remaining_cards[j] for j in np.random.choice(len(self.remaining_cards), n, replace=False)]
        [self.remaining_cards.remove(j) for j in cards]
        return cards

    def count_pairs(self, state):
        c = Counter([i[0] for i in state])

        pairs_list = [i for i in c if c[i] >= 2]
        three_kind_list = [i for i in c if c[i] >= 3]
        # print(pairs_list)
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
        # three_kind_list = [i for i in c if c[i] >= 3]

    def count_flush(self, state):
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

    def cards_in_row(self, state):
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

    def fill_in_state(self, round):
        for i in self.player_list:
            if round == 'opening_state':
                full_state, full_rank = self.find_state(i.curr_hand)
                i.update_dict[round] = full_state
                continue
            full_state, full_rank = self.find_state(i.curr_hand + self.table_cards)
            table_state, table_rank = self.find_state(self.table_cards)
            table_state[0] = 2 if table_state[0] == full_state[0] else 1
            i.curr_state = full_state + table_state
            i.curr_rank = full_rank
            i.update_dict[round] = full_state + table_state

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

    def run_multiple(self, n):

        # TODO Time everything in here and find bottlenecks
        for i in range(n):
            # if i % 10000 == 0:
            #     with open('update_dict.pkl', 'wb') as f:
            #         pickle.dump(self.stat_dict, f)
            result, result_player_list = self.run_game()
            for player in self.player_list:
                for state in player.update_dict.keys():
                    player.update_dict[state] = player.update_dict[state] + [1, 0]
            for player in result_player_list:
                # TODO Replace all this with dictionary {final_state:[..]..etc}
                for state in player.update_dict.keys():
                    player.update_dict[state][-1] = 1
            for state in self.stat_dict.keys():
                self.stat_dict[state] += [player.update_dict[state] for player in self.player_list]

        for state in self.stat_dict.keys():
            file_name = '{}_df.csv'.format(state)
            state_array = np.array(self.stat_dict[state])
            state_df = pd.DataFrame(state_array)
            if state == 'opening_state':
                state_groupby = state_df.groupby([0, 1, 2, 3, 4,]).sum().reset_index()
                print(state_groupby.shape)
                state_groupby.to_csv(file_name)
                continue
            print(state_df)
            state_groupby = state_df.groupby([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).sum().reset_index()
            print(state_groupby.shape)
            state_groupby.to_csv(file_name)
        return self.stat_dict


if __name__ == '__main__':
    pkr = PokerAgent(5)
    # print(pkr.find_state([(3, 'h'), (4, 'h'), (2, 'h'), (1, 'h'), (6, 'h'), (10, 'h'), (5, 'h')]))
    a = pkr.run_multiple(1000000)
    # print(a['final_state'])
# print(a_a)
# print(a_a.shape)
# print(np.array(a))
# unique_a = np.unique(a_a[:, 0:10], axis=0)
# # print(unique_a)
# roll_up = []
# for i in unique_a:
#     roll_up.append(np.concatenate([i , a_a[np.all(a_a[:, 0:10] == i, axis=1), -2:].sum(axis=0)], axis = 0))
# np.array(roll_up)
# print(unique_a.shape)

# print(a)
# print(a, b[0].curr_state)
# for j in pkr.player_list:
#     state = j.curr_hand + pkr.table_cards
#     print(state)
#     print(j.curr_state)
#     print(j.curr_rank)
# print(pkr.count_hands(pkr.table_cards))
