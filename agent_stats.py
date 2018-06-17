import numpy as np
import pandas as pd
from PokerBot.agent_base import PokerAgentBase


class PokerAgentStats(PokerAgentBase):
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
                state_groupby = state_df.groupby([0, 1, 2, 3, 4, ]).sum().reset_index()
                print(state_groupby.shape)
                state_groupby.to_csv(file_name)
                continue
            print(state_df)
            state_groupby = state_df.groupby([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).sum().reset_index()
            print(state_groupby.shape)
            state_groupby.to_csv(file_name)
        return self.stat_dict


if __name__ == '__main__':
    pkr = PokerAgentStats(5)
    a = pkr.run_multiple(1000000)
