import config
from game import *
from network import *
from replaybuffer import *

import torch
import time
import ray
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MuZero():
    ray.init()
    network = Network()
    replaybuffer = RepalyBuffer()

    for i in range(config.num_iters):
        run_selfplay(network.to('cpu'), replaybuffer)
        train_network()
        
        if i % config.test_interval:
            test_play()
    
    ray.shutdown()
    print('COMPLATED')
        


def run_selfplay(network, replaybuffer):
    start = time.time()
    work_in_progresses = [selfplay.remote(network) for i in range(15)]

    for i in range(config.num_selfplay):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        replaybuffer.save_game(ray.get(finished)[0])
        work_in_progresses.extend([selfplay.remote(network)])
        print(f'{int(i*100/config.num_selfplay)}% done')#!
    print(f'Elapsed(run_selfplay):{time.time() - start}')

@ray.remote
def selfplay(network):
    game = Game()
    while not game.is_terminated and len(game.history['action']) < config.max_moves:
        root = Node(0, game.to_play())
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            # print(current_observation.size())
            net_output = network.initial_inference(current_observation) #! on cpu
        game.is_terminated=True #!
    return game

def train_network():
    pass

def test_play():
    pass

if __name__ == '__main__':
    MuZero()