import numpy as np
import pandas as pd
from PokerBot.agent_base import PokerAgentBase


class PokerAgent(PokerAgentBase):
    def player_actions(self):
        remove_list = []
        for player in self.player_list:
            action, bet = player.decide_action(game_state=self.game_state, big_blind=self.big_blind,
                                               curr_table=self.curr_pot, curr_max_bet=self.curr_max_bet)
            if action == 'fold':
                remove_list.append(player)
                if len(self.player_list) - len(remove_list) == 1:
                    break
                continue
            #TODO this should all be on player?
            # player.curr_money -= bet
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
        for player in self.starting_player_list:
            player.start_money = player.curr_money
        self.player_list = self.starting_player_list
        self.curr_pot = 0
        self.curr_pot = self.big_blind + self.small_blind
        self.curr_max_bet = self.big_blind
        self.player_list = self.player_list[self.dealer + 2:] + self.player_list[:self.dealer + 2]
        self.player_list[-1].curr_bet = self.big_blind
        self.player_list[-1].curr_money -= self.big_blind
        self.player_list[-2].curr_bet = self.small_blind
        self.player_list[-2].curr_money -= self.small_blind

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
            self.curr_pot -= (max_bet - sec_bet)
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

    def run_tournament(self):
        hand_counter = 1
        while len(self.starting_player_list) > 1:
            print(hand_counter)
            self.run_game()
            for player in self.starting_player_list:
                print(player.curr_money, player.curr_bet)
                if player.curr_money == 0:
                    self.starting_player_list.remove(player)
            if hand_counter % 100 == 0:
                self.big_blind *= 2
                self.small_blind *= 2
            hand_counter += 1
        print('finished')
        print(self.starting_player_list[0].curr_money, self.starting_player_list[0].name)

if __name__ == '__main__':
    pkr = PokerAgent(5)
    a = pkr.run_tournament()
    # print(a)
