import config
from game import *
from network import *
from replaybuffer import *
from mcts import *
import torch
import torch.nn as nn
import time
import ray
from tqdm import trange
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MuZero():
    def __init__(self):
        ray.init()
        self.network = Network()
        self.replaybuffer = RepalyBuffer()
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.v_loss_fn = nn.MSELoss()
        self.p_loss_fn = nn.CrossEntropyLoss()
        self.total_losses = []
        self.v_losses = []
        self.p_losses = []
        self.win_rate = []
        self.SharedStorage = deque(maxlen=2)

    def run(self):
        start = time.time()

        for i in range(config.num_iters):
            self.run_selfplay(i)
            self.train_network(i)
            if i <= 20:
                self.SharedStorage.append(copy.deepcopy(self.network.state_dict()))

            self.plot_each_loss()
            self.plot_total_loss()

            elapsed = time.time() - start
            print(f'Total:{elapsed//3600}h{(elapsed%3600)//60}m{(elapsed%60)}s ')

            if i % config.test_interval ==  0 and i > 19:
                self.SharedStorage.append(copy.deepcopy(self.network.state_dict()))
                self.run_testplay(i)
                self.plot_winning_rate()

        ray.shutdown()
        print('COMPLATED')


    def run_selfplay(self, j):
        self.network.to('cpu')
        start = time.time()
        work_in_progresses = [selfplay.remote(self.network) for i in range(15)]

        for i in trange(config.num_selfplay):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            self.replaybuffer.save_game(ray.get(finished)[0])
            work_in_progresses.extend([selfplay.remote(self.network)])

        print(f'{j}th run_selfplay:{time.time() - start}sec |', end='')

    def train_network(self, j):
        self.network.to(device)

        start = time.time()

        self.network.train()
        self.network.to(device)

        for i in range(config.num_epoch):

            batch = self.replaybuffer.sample_batch()
            self.update_weights(batch)

        print(f'{j}th train_network:{time.time() - start}sec |', end='')
        self.network.to('cpu')


    def update_weights(self, batch):

        self.optimizer.zero_grad()
        v_loss, p_loss = 0, 0

        encoded_states, hidden_states, encoded_actions, target_values, target_policies = batch
        encoded_states, hidden_states, encoded_actions, target_values, target_policies = torch.FloatTensor(encoded_states).to(device).to(device), torch.FloatTensor(hidden_states).to(device), torch.FloatTensor(encoded_actions).to(device), torch.FloatTensor(target_values).to(device), torch.FloatTensor(target_policies).to(device)

        values_ini, policies_ini, hidden_states_ini = self.network.initial_inference(encoded_states)
        values_rec, policies_rec = self.network.prediction(hidden_states)

        #torch.mean((target_values - values_ini) ** 2) + torch.mean((target_values - values_rec) ** 2)
        v_loss = self.v_loss_fn(target_values, values_ini) + self.v_loss_fn(target_values, values_rec)


        # for target, logit in zip(target_policies, policies_ini):
        #     p_loss += nn.CrossEntropyLoss(target, logit)

        # for target, logit in zip(target_policies, policies_rec):
        #     p_loss += nn.CrossEntropyLoss(target, logit)

        #p_loss /= config.batch_size

        p_loss = self.p_loss_fn(target_policies, policies_ini) + self.p_loss_fn(target_policies, policies_rec)

        total_loss = (p_loss + v_loss)
        total_loss.backward()
        self.optimizer.step()

        self.total_losses.append(total_loss.detach().cpu().numpy())
        self.v_losses.append(v_loss.detach().cpu().numpy())
        self.p_losses.append(p_loss.detach().cpu().numpy())

    def run_testplay(self, j):

        work_in_progresses = [testplay.remote(self.SharedStorage) for i in range(5)]
        result = []
        for i in range(config.num_testplay):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            result.append(ray.get(finished)[0])
            work_in_progresses.extend([testplay.remote(self.SharedStorage)])

        zero_score = np.sum(x==0 for x in result)
        one_score = np.sum(x==1 for x in result)

        self.win_rate.append(one_score/config.num_testplay)

        if zero_score > one_score:
            print('newest model lose')
            self.SharedStorage.append(self.SharedStorage[0])
        else:
            print('newest model win')
            self.SharedStorage.append(self.SharedStorage[1])

        torch.save(self.SharedStorage[1], f'../models/model{j}.pth')

    def plot_each_loss(self):

        plt.plot(range(len(self.total_losses)), self.total_losses, label='total loss')
        plt.plot(range(len(self.v_losses)), self.v_losses, label='value_loss')
        plt.plot(range(len(self.p_losses)), self.p_losses, label='policy_loss')
        plt.legend()
        plt.xlabel('updates')
        plt.ylabel('Loss')
        plt.savefig(f'../out/each_loss.pdf')
        plt.clf()

    def plot_total_loss(self):

        plt.plot(range(len(self.total_losses)), self.total_losses, label='total loss')
        plt.legend()
        plt.xlabel('updates')
        plt.ylabel('Loss')
        plt.savefig(f'../out/learning_curve.pdf')
        plt.clf()

    def plot_winning_rate(self):
        plt.plot(range(len(self.win_rate)), self.win_rate)
        plt.xlabel('updates')
        plt.ylabel('winning rate')
        plt.savefig(f'../out/winning_rate.pdf')
        plt.clf()



@ray.remote
def selfplay(network):
    game = Game()
    while not game.is_terminated() and len(game.history['action']) < config.max_moves:
        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network.initial_inference(current_observation) #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)


    time.sleep(1)

    return game


def closs_entropy(target, logit):
    loss = 0
    for t, p in  zip(target, logit):
        loss += -t*torch.log(p+1e-8)
    return loss


@ray.remote
def testplay(SharedStorage):
    network1 = Network()
    network2 = Network()
    a = random.randint(0, 1)
    network1.load_state_dict(SharedStorage[a])
    network2.load_state_dict(SharedStorage[1-a])

    game = Game()
    while not game.is_terminated() and len(game.history['action']) < config.max_moves:
        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network1.initial_inference(current_observation) #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network1)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)

        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network2.initial_inference(current_observation) #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network2)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)

    winner = game.environment.check_win()

    if winner == 1:
        return a
    else:
        return 1-a


def test_selfplay():
    game = Game()
    network = Network()
    while not game.is_terminated() and len(game.history['action']) < config.max_moves:
        game.environment.print_board()
        print([a.position for a in game.legal_actions()])
        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network.initial_inference(current_observation) #! on cpu
        root.expand_node(game.legal_actions(), net_output)
        
        action = MCTS(root, game, network)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)

    game.environment.print_board()
    game.environment.check_win2()
    target_value, target_policy = game.get_target(55)

    print(f'target_value:{target_value}')
    print(f'target_policy:{target_policy}')

def test_rb(replaybuffer):
    encoded_state, hidden_state, encoded_actions, target_values, target_policies = replaybuffer.sample_batch()
    print(f'encoded_state:{init_state.shape}')
    print(f'encoded_actions:{encoded_actions.shape}')
    print(f'target_values:{target_values.shape}')
    print(f'target_policies:{target_policies.shape}')

if __name__ == '__main__':
    #_test()
    muzero = MuZero()
    muzero.run()