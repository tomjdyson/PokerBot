import numpy as np
from collections import Counter
import pickle
import pandas as pd
from player_bet import SimpleBet


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

        self.betting_obj = SimpleBet(np.random.randint(-20, 0) / 10, np.random.randint(20, 40) / 10,
                                     np.random.randint(0, 40) / 10)
        # call_risk AND raise_risk

    def reset_state(self):
        self.curr_hand = None
        self.curr_bet = None
        self.curr_state = None
        self.curr_rank = None

    def decide_action(self, game_state, big_blind, curr_max_bet, curr_table):
        state_df = self.stat_dict[game_state].values

        stat_array = state_df[np.all(state_df[:, :-2] == np.array(self.curr_state), axis=1), -2:][0]
        risk = stat_array[1] / stat_array[0]

        self.previous_action, bet = self.betting_obj.action(risk=risk, big_blind=big_blind, curr_max_bet=curr_max_bet,
                                                            curr_table=curr_table, curr_self_bet=self.curr_bet)
        # TOdo dont like how this is done,  shouldnt return should jsut read bet
        self.curr_bet += bet

        if self.curr_bet >= self.curr_money:
            bet = self.curr_money
        return self.previous_action, bet


class PokerAgent:
    def __init__(self, num_players):
        self.player_1 = PokerPlayer()
        self.player_2 = PokerPlayer()
        self.player_3 = PokerPlayer()
        self.player_4 = PokerPlayer()
        self.player_5 = PokerPlayer()
        self.player_list = [self.player_1, self.player_2, self.player_3, self.player_4, self.player_5][0:num_players]
        self.starting_player_list = self.player_list
        self.dealer = np.random.randint(0, num_players)
        self.small_blind = 1
        self.big_blind = 2
        self.curr_cards = None
        self.remaining_cards = None
        self.all_cards = self.create_cards()
        self.table_cards = None
        self.curr_pot = self.big_blind + self.small_blind
        self.curr_max_bet = 0
        self.game_state = None

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

    def fill_in_state(self, curr_round):
        self.game_state = curr_round
        for i in self.player_list:
            if curr_round == 'opening_state':
                full_state, full_rank = self.find_state(i.curr_hand)
                i.update_dict[curr_round] = full_state
                i.curr_state = full_state
                continue
            full_state, full_rank = self.find_state(i.curr_hand + self.table_cards)
            table_state, table_rank = self.find_state(self.table_cards)
            table_state[0] = 2 if table_state[0] == full_state[0] else 1
            i.curr_state = full_state + table_state
            i.curr_rank = full_rank
            i.update_dict[curr_round] = full_state + table_state

    def player_actions(self):
        remove_list = []
        for player in self.player_list:
            print(player.name)
            action, bet = player.decide_action(game_state=self.game_state, big_blind=self.big_blind,
                                               curr_table=self.curr_pot, curr_max_bet=self.curr_max_bet)
            print(action, bet, player.curr_bet, self.curr_max_bet)
            if action == 'fold':
                remove_list.append(player)
                if len(self.player_list) - len(remove_list) == 1:
                    print('breaking')
                    break
                continue
            player.curr_money -= bet
            self.curr_pot += bet
            self.curr_max_bet = max(self.curr_max_bet, player.curr_bet)
        [self.player_list.remove(player) for player in remove_list]

    def all_players_bet(self):
        unequal = False
        for player in self.player_list:
            if player.curr_bet == self.curr_max_bet:
                unequal = False
                continue
            elif player.curr_bet >= player.curr_money:
                unequal = False
                continue
            else:
                unequal = True
                break
        return unequal

    def initialize_game(self):
        self.player_list = self.starting_player_list
        self.curr_pot = 0
        self.curr_pot = self.big_blind + self.small_blind
        self.curr_max_bet = self.big_blind
        self.player_list = self.player_list[self.dealer + 2:] + self.player_list[:self.dealer + 2]
        self.player_list[-1].curr_bet = self.big_blind
        self.player_list[-2].curr_bet = self.small_blind

    def run_betting(self):
        print(len(self.player_list))
        if len(self.player_list) > 1:
            unequal = True
            while unequal:
                self.player_actions()
                unequal = self.all_players_bet()

    def handle_money(self, result, player_list):
        if result == 'draw':
            money_amount = self.curr_pot / len(player_list)
            for player in player_list:
                player.curr_money += money_amount
        else:
            player_list[0].curr_money += self.curr_pot

    def run_game(self):
        self.initialize_game()
        self.remaining_cards = self.create_cards()
        for i in self.player_list:
            hand = self.pick_cards(2)
            i.curr_hand = hand
        self.fill_in_state('opening_state')
        # while loop if all players not on same amount - stick in betting so not infinite loop
        # insert bet
        self.run_betting()
        self.table_cards = self.pick_cards(3)
        self.fill_in_state('flop_state')
        # insert bet
        self.run_betting()
        self.table_cards += self.pick_cards(1)
        self.fill_in_state('turn_state')
        # insert bet
        self.run_betting()
        self.table_cards += self.pick_cards(1)
        # insert_bet
        self.fill_in_state('final_state')
        self.run_betting()
        result, player_list = self.find_best()
        self.handle_money(result, player_list)
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


if __name__ == '__main__':
    # player = PokerPlayer()
    # player.curr_state = [1, 3, 2, 1, 1]
    # player.decide_action('opening_state')
    pkr = PokerAgent(5)
    a = pkr.run_game()
    print(a)
    # print(pkr.find_state([(3, 'h'), (4, 'h'), (2, 'h'), (1, 'h'), (6, 'h'), (10, 'h'), (5, 'h')]))
    # a = pkr.run_multiple(1000000)
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
