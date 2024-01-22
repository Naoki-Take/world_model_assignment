import config

from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import math


class Player(Enum):
    BLACK = 1
    WHITE = -1

def get_opponent_player(player):
    player = Player.WHITE if player is Player.BLACK else Player.BLACK
    return player

@dataclass
class Action:
    index: int
    player: Player # check leggal position mathces for action player
    # position: Optional[Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        assert self.index >= 0 and self.index <= config.board_length**2
        if self.index < config.board_length**2:
            self.position = divmod(self.index, 8)
        else:
            self.position = None
    
    def get_encoded_action(self):
        encoded_action = np.zeros((2, config.board_length, config.board_length), dtype=np.bool)
        layer = 0 if self.player is Player.BLACK else 1
        encoded_action[layer, self.position] = 1
        return encoded_action


DIRECTIONS = (np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]),
            np.array([0, 1]), np.array([1, -1]), np.array([1, 0]), np.array([1, 1]))
    

class Environment:
    def __init__(self):
        self.board = np.zeros((config.board_length, config.board_length), dtype=np.int8)
        a = config.board_length//2 - 1
        b = config.board_length//2
        self.board[a, b] = 1
        self.board[b, a] = 1
        self.board[a, a] = -1
        self.board[b, b] = -1
        

        self.possible_pos = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5),
                            (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5,5)}

        self.legal_actions = [Action(2*8+3, Player.BLACK), Action(3*8+2, Player.BLACK), Action(5*8+4, Player.BLACK), Action(4*8+5, Player.BLACK)]

        self.player = Player.BLACK

        self.steps = 0

        self.was_pass = False
        self.done = False

    def check_direction(self, position, direction):
        next_position = position + direction
        step = 0

        while np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)] == -self.player.value:
            step += 1
            next_position += direction

        if step != 0 and np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)] == self.player.value:
            return True
        else:
            return False
    
    def check_position(self, position):
        position = np.array(position)

        for direction in DIRECTIONS:
            if self.check_direction(position, direction):
                # print('{}, {}, {}'.format(position, direction, next_mark))
                return True
        
        return False

    def update_legal_actions(self):
        self.legal_actions = []
        for position in self.possible_pos:
            if self.check_position(position):
                self.legal_actions.append(Action(position[0]*8+position[1], self.player))
        # add pass as action[64]
        if not self.legal_actions:
            self.legal_actions.append(Action(64, self.player))
    
    def update_possible_position(self, position):
        self.possible_pos.remove(position)

        for direction in DIRECTIONS:
            around_position = position + direction
            if np.all(around_position>=0) and np.all(around_position<=7) and self.board[tuple(around_position)] == 0:
                self.possible_pos.add(tuple(around_position))
    
    def step(self, action):

        assert action.player is self.player
        assert action in self.legal_actions, '{} out of {}'.format(action, self.legal_actions)

        self.steps += 1

        # process the pass
        if action.index == 64:
            if self.was_pass == True:
                self.done = True
                return self.observe(), self.check_win()
            else:
                self.was_pass = True
                self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK
                self.update_legal_actions()
                return self.observe(), 0

        self.was_pass = False

        position = np.array(action.position)

        opponent_player = get_opponent_player(self.player)

        # flip the pieces
        for direction in DIRECTIONS:

            next_position = position + direction
            step = 0

            while np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)]==opponent_player.value:
                next_position = next_position + direction
                step += 1

            if step != 0 and np.all(next_position>=0) and np.all(next_position<=7) and self.board[tuple(next_position)] == self.player.value:
                for _ in range(step):
                    next_position = next_position - direction
                    self.board[tuple(next_position)] = self.player.value

        # put a disc
        self.board[action.position] = self.player.value

        # board full
        if self.steps == 60:
            self.done = True
            return self.observe(), self.check_win()


        self.update_possible_position(action.position)
        self.player = Player.WHITE if self.player is Player.BLACK else Player.BLACK
        self.update_legal_actions()

        return self.observe(), 0
    
    def check_win(self):

        black_score = np.sum(self.board==1)
        white_score = np.sum(self.board==-1)
        
        if black_score == white_score:
            return 0
        elif black_score > white_score:
            return self.player.BLACK.value
        else:
            return self.player.WHITE.value

    def check_win2(self):

        black_score = np.sum(self.board==1)
        white_score = np.sum(self.board==-1)
        print(f'BLACK:{black_score}')
        print(f'WHITE:{white_score}')
        if black_score == white_score:
            print('DROW')
            return None
        elif black_score > white_score:
            print('BLACK WIN')
            return None
        else:
            print('WHITE_WIN')
            return None
    
    def observe(self):

        obs = np.stack((self.board==Player.WHITE.value, self.board==Player.BLACK.value), axis=0).astype(bool)

        return obs
    
    def print_board(self):
        char_board = np.vectorize(index_to_char)(self.board)
        print("  0  1  2  3  4  5  6  7")
        for i in range(8):
            print(i, end='')
            for j in range(8):
                print(f' {char_board[i][j]} ', end='')
            print('')

def index_to_char(index):
    if index == 1:
        return '●'
    elif index == 0:
        return '-'
    elif index == -1:
        return '○'


class Node:
    def __init__(self, prior, parent = None):
        self.hidden_state = None
        self.reward = 0

        self.prior = prior
        self.value_sum = 0
        self.visit_count = 0

        self.parent = parent
        self.children = {}

    def is_expanded(self):

        return len(self.children) > 0
    
    def value(self):

        if self.visit_count == 0:
            return 0
        
        return self.value_sum / self.visit_count

    def expand_node(self, actions, network_output):
        self.hidden_state = network_output[2].squeeze()
        policy = {a.index: network_output[1][0, a.index].item() for a in actions}
        for action, p in policy.items():
            self.children[action] = Node(prior=p, parent=self)
    
    def select_child(self):

        _, action, child = max((self.ucb_score(child), action, child) for action, child in self.children.items())
        return action, child
    
    def ucb_score(self, child):

        pb_c = math.log((self.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(self.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        
        return child.value() + prior_score
    
    def backpropagate(self, value: float):

        self.value_sum += value
        self.visit_count += 1

        value = self.reward + config.discount * (-value)
        if self.parent is not None:
            self.parent.backpropagate(value)



class Game:
    def __init__(self):

        self.environment = Environment()
        self.history = {'action':[], 'state':[self.environment.observe()], 'reward':[]}
    
    def get_encoded_state(self, state_idx):

        encoded_state = np.zeros(config.state_shape, dtype=bool) #! why bool
        for i in range(min(state_idx+1, config.state_history_len)):
            encoded_state[(config.state_history_len-1-i)*2:(config.state_history_len-i)*2] = self.history['state'][state_idx-i]
        if state_idx % 2 == 1:
            encoded_state[-1] = 1

        return encoded_state.astype(np.float32)
    
    def legal_actions(self):

        return self.environment.legal_actions
    
    def apply(self, action: Action):

        state, reward = self.environment.step(action)

        self.history['state'].append(state)
        self.history['action'].append(action)
        self.history['reward'].append(reward)

    def is_terminated(self):
        return self.environment.done

    def to_play(self):

        return self.environment.player

def test_PvP():
    # PvP
    game = Game()
    game.environment.print_board()
    while True:
        legal_actions = game.legal_actions()
        for i, action in enumerate(legal_actions):
            print(f'{i}:{action.position}', end="")
        print('')
        is_invalid_action = True
        while is_invalid_action:
            action_idx = int(input("Your Turn:"))
            if action_idx in range(len(legal_actions)):
                is_invalid_action = False
            else:
                print(f'{action_idx} is invalid action')
        
        game.apply(legal_actions[action_idx])
        game.environment.print_board()

        if game.is_terminated():
            game.environment.check_win2()
            break


def test_Node():
    pass

if __name__ == '__main__':
    test_PvP()