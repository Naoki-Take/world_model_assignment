from enum import Enum
import numpy as np
import config

class Player(Enum):
    BLACK = 1
    WHITE = -1
    

class Environment:
    def __init__(self):
        self.board = np.zeros((config.board_length, config.board_length), dtype=np.int8)
        a = config.board_length//2 - 1
        b = config.board_length//2
        self.board[a, a] = 1
        self.board[b, b] = 1
        self.board[a, b] = -1
        self.board[b, a] = -1
        self.player = Player.BLACK
        self.steps = 0
    
    def observe(self):
        obs = np.stack((self.board==Player.WHITE.value, self.board==Player.BLACK.value), axis=0).astype(np.bool)

        return obs

class Node:
    def __init__(self, player, parent=None, action_taken=None):
        self.player = player
        self.parent = parent
        self.action_taken = action_taken

class Game:
    def __init__(self):
        self.environment = Environment()
        self.is_terminated = False
        self.history = {'action':[], 'state':[self.environment.observe()]}
        
    def apply(self, action):
        self.is_terminated = True # or else
        pass
    
    def get_encoded_state(self, state_idx):
        encoded_state = np.zeros(config.state_shape, dtype=bool) #! why bool
        for i in range(min(state_idx+1, config.state_history_len)):
            encoded_state[(config.state_history_len-1-i)*2:(config.state_history_len-i)*2] = self.history['state'][state_idx-i]
        if state_idx % 2 == 1:
            encoded_state[-1] = 1

        return encoded_state.astype(np.float32)

    def to_play(self) -> Player:
        return self.environment.player

if __name__ == '__main__':
    game = Game()
    print(game.get_encoded_state(0))
    print(game.to_play())