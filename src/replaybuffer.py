import config

import numpy as np
import random
from collections import defaultdict, deque


class RepalyBuffer:
    def __init__(self):
        self.batch_size = config.batch_size
        self.buffer = deque([], maxlen=config.window_size)

    def save_game(self, game):
        self.buffer.append(game)

    def sample_batch(self):#!

        game_idxes = random.sample(range(len(self.buffer)), self.batch_size)
        games = [self.buffer[idx] for idx in game_idxes]
        game_pos = [(g, self.sample_position(g)) for g in games]
        # batch = [(g.make_image(i), g.history[i:i + num_unroll_steps],
        #         g.make_target(i, num_unroll_steps, td_steps))
        #         for (g, i) in game_pos]

        encoded_states = []
        hidden_states = []
        encoded_actions = []
        target_values = []
        target_policies = []
        for g, i in game_pos:
            #print(f'g.get_encoded_state(i):{g.get_encoded_state(i)}')
            encoded_states.append(g.get_encoded_state(i))
            #print(f"g.history['hidden_state'][i]:{np.array(g.history['hidden_state'][i])}")
            hidden_states.append(np.array(g.history['hidden_state'][i]))
            #print(f"g.history['action'][i].get_encoded_action():{g.history['action'][i].get_encoded_action()}")
            encoded_actions.append(g.history['action'][i].get_encoded_action())
            #encoded_actions.append(np.array([ a.get_encode_action() for a in g.history['action'][i:i + num_unroll_steps]]))
            target_value, target_policy = g.get_target(i)
            #print(f'target_value:{target_value}')
            #print(f'target_policy:{target_policy}')
            target_values.append([target_value])
            target_policies.append(target_policy)

        return np.array(encoded_states), np.array(hidden_states), np.array(encoded_actions), np.array(target_values), np.array(target_policies)
    
    def sample_position(self, game) -> int:

        return random.randint(0, len(game.history['action'])-config.num_unroll_steps)


    
    def get_len(self):
        return len(self.buffer)
    
    