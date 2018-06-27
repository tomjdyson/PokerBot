import numpy as np
import pandas as pd
from PokerBot.agent_base import PokerAgentBase
from collections import Counter
from PokerBot.bet_model import RFModel


# TODO Check out of list error - fixed with try for now but need to simulate again probably 10m
# TODO Check blinds - seem to not be removing money , also not swapping with 3 players
# TODO Side pots
# TODO Change single max bet to single max raise and make it work
# TODO shuffle starting list on initialize

class PokerAgent(PokerAgentBase):
    def player_actions(self):
        remove_list = []
        for player in self.player_list:
            action, bet = player.decide_action(game_state=self.game_state, big_blind=self.big_blind,
                                               curr_table=self.curr_pot, curr_max_bet=self.curr_max_bet,
                                               remaining_player_hands=len(self.player_list),
                                               remaining_players_tournament=len(self.starting_player_list),
                                               hand_lowest_money=min(
                                                   [player.start_money for player in self.player_list]),
                                               single_max_raise=self.single_max_raise)
            if player.curr_money < 0:
                raise (ValueError, 'more than money')
            # print('name', player.name, 'curr_money', player.curr_money, 'start_money', player.start_money, 'action',
            #       action, 'bet', bet, 'curr_bet',
            #       player.curr_bet, 'max_bet', self.curr_max_bet, 'curr_pot', self.curr_pot)
            if action == 'fold':
                remove_list.append(player)
                if len(self.player_list) - len(remove_list) == 1:
                    break
                continue
            # TODO this should all be on player?
            # player.curr_money -= bet
            self.curr_pot += bet
            self.single_max_raise = max(self.single_max_raise, self.curr_max_bet - player.curr_bet)
            self.curr_max_bet = max(self.curr_max_bet, player.curr_bet)
            if player.curr_bet > player.start_money:
                raise (ValueError, 'more than money')
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
        for player in self.starting_player_list:
            if player.curr_money < 0:
                raise (ValueError, 'in debt')
            player.start_money = player.curr_money
            player.curr_bet = 0
        self.player_list = self.starting_player_list
        self.curr_pot = 0
        self.curr_pot = self.big_blind + self.small_blind
        # self.curr_max_bet = self.big_blind
        self.single_max_raise = self.big_blind
        # Cant handle when someone leaves tournament but is fine for now
        if self.dealer >= len(self.starting_player_list) - 1:
            self.dealer = 0
        else:
            self.dealer += 1
        if len(self.player_list) > 2:
            big_blind_pos = self.dealer - (len(self.player_list) - 2) if self.dealer >= (len(self.player_list) - 2) else self.dealer + 2
            small_blind_pos = self.dealer - (len(self.player_list) - 1) if self.dealer >= (len(self.player_list) - 1) else self.dealer + 1
            self.player_list = self.starting_player_list[big_blind_pos + 1:] + self.starting_player_list[
                                                                               :big_blind_pos + 1]

        else:
            small_blind_pos = self.dealer
            big_blind_pos = (self.dealer + 1) % 2
            # TODO This might not work - need to look at heads up rules who starts betting
            self.player_list = [self.starting_player_list[small_blind_pos]] + [self.starting_player_list[big_blind_pos]]

        self.starting_player_list[big_blind_pos].curr_bet = min(self.big_blind,
                                                                self.starting_player_list[big_blind_pos].start_money)
        self.starting_player_list[big_blind_pos].curr_money -= min(self.big_blind,
                                                                   self.starting_player_list[big_blind_pos].start_money)
        self.starting_player_list[small_blind_pos].curr_bet = min(self.small_blind, self.starting_player_list[
            small_blind_pos].start_money)
        self.starting_player_list[small_blind_pos].curr_money -= min(self.small_blind, self.starting_player_list[
            small_blind_pos].start_money)

        self.curr_max_bet = max(player.curr_bet for player in self.starting_player_list)

        # print('Starting location:' , [player.name for player in self.starting_player_list])
        # print('Dealer Location:',[player.name for player in self.player_list], big_blind_pos, small_blind_pos)

    def run_betting(self):
        if len(self.player_list) > 1:
            unequal = True
            while unequal:
                self.player_actions()
                unequal = self.all_players_bet()

    def side_pool(self):
        player_bets = [player.curr_bet for player in self.player_list]
        max_bet = max(player_bets)
        m_l = [i for i, j in enumerate(player_bets) if j == max_bet]
        player_bets.remove(max_bet)
        sec_bet = max(player_bets)
        ranks = [i.curr_rank for i in self.player_list]
        ranks_sort = ranks.copy()
        ranks_sort.sort()
        if len(m_l) == 1:
            self.player_list[m_l[0]].curr_money += (max_bet - sec_bet)
            self.curr_pot = self.curr_pot - (max_bet - sec_bet)
            self.player_list[m_l[0]].curr_bet = sec_bet
            ###TODO Finish this off - too complicated for now
            # curr_rank = 0
            # while self.curr_pot > 0:
            #     rank = ranks_sort[curr_rank]
            #     print(rank)
            #     print(self.curr_pot)
            #     rank_pos = [i for i, j in enumerate(ranks) if j == rank]
            #     if len(rank_pos) == 1:
            #         curr_winner = self.player_list[rank_pos[0]]
            #         pot_money = len(self.player_list) * curr_winner.curr_bet
            #         if pot_money > self.curr_pot:
            #             pot_money = self.curr_pot
            #         print(pot_money, curr_winner.curr_bet)
            #         curr_winner.curr_money += pot_money
            #         self.curr_pot -= pot_money
            #     else:
            #         # print(rank_pos)
            #         for i in rank_pos:
            #             # print(i, self.player_list)
            #             curr_winner = self.player_list[i]
            #             winner_money = len(self.player_list) * curr_winner.curr_bet
            #             if winner_money > self.curr_pot:
            #                 winner_money = self.curr_pot
            #             print(winner_money)
            #             curr_winner.curr_money += winner_money / len(rank_pos)
            #             self.curr_pot -= winner_money / len(rank_pos)
            #     curr_rank += 1

    def handle_money(self, result, player_list):

        all_bets = [player.curr_bet for player in self.player_list]
        if len(set(all_bets)) != 1:
            self.side_pool()

        if result == 'draw':
            money_amount = round(self.curr_pot / len(player_list))
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

    def run_tournament(self):
        hand_counter = 1
        while len(self.starting_player_list) > 1:
            # print(hand/_counter)
            self.run_game()
            for player in self.starting_player_list:
                # print(player.name, player.curr_money, player.curr_bet, self.curr_pot)
                if player.curr_money == 0:
                    self.starting_player_list.remove(player)
                if player.curr_money < 0:
                    ValueError('Cant be in debt')
            if hand_counter % 50 == 0:
                self.big_blind *= 2
                self.small_blind *= 2
            hand_counter += 1

        # print('finished')
        # print(self.starting_player_list[0].curr_money, self.starting_player_list[0].name,
        #       self.starting_player_list[0].tournament_hands)
        # print(self.starting_player_list[0].name)
        return self.starting_player_list


def simulate_tournaments(n):
    simulate = []
    winners = []
    rf_clf = RFModel('C:/Users/dysont/Documents/Graduate/rl/PokerBot/tournament_data_2.csv', 1000000)
    action_clf = rf_clf.action_model()
    bet_clf = rf_clf.bet_model()
    for i in range(n):
        pkr = PokerAgent(5, action_clf=action_clf, bet_clf=bet_clf)
        winner = pkr.run_tournament()[0]
        simulate.append(winner.tournament_hands)
        winners.append(winner.name)
        # print(i)
    # TODO Better way for this
    flat_list = [item for sublist in simulate for item in sublist]
    df = pd.DataFrame(flat_list)
    print(Counter(winners))

    return df


if __name__ == '__main__':
    from multiprocessing import Pool

    # game_obj = PokerAgent(5)
    # print(game_obj.run_tournament())
    # for i in [1, 2, 3]:
    simulate_tournaments(10000).to_csv('tournament_data_3.csv')

    # with Pool(3) as p:
    #     a, b, c = p.map(simulate_tournaments, [500, 500, 500])
    #
    # pd.concat([a, b, c], axis=0).to_csv('tournament_data.csv')



    # simulate_tournaments(10).to_csv('tournament_data.csv')
    # print(a)
