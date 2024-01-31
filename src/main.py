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
from matplotlib.ticker import FixedLocator
from collections import defaultdict, deque
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MuZero():
    def __init__(self):
        ray.init()
        self.network = Network()
        self.replaybuffer = RepalyBuffer()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.v_loss_fn = nn.MSELoss()
        self.p_loss_fn = nn.CrossEntropyLoss()
        self.total_losses = []
        self.v_losses = []
        self.p_losses = []
        self.vae_losses = []
        self.win_rate = []
        self.SharedStorage = deque(maxlen=2)

    def run(self):
        start = time.time()

        for i in range(config.num_iters):
            self.run_selfplay(i)
            self.train_network(i)
            if i <= 20:
                state = {
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(state, '../var/champion.pth')

            self.plot_each_loss()
            self.plot_total_loss()
            self.plot_losses()

            elapsed = time.time() - start
            print(f'Total:{elapsed//3600}h{(elapsed%3600)//60}m{(elapsed%60)}s ')

            if i % config.test_interval ==  0 and i > 15:
                state = {
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(state, '../var/challenger.pth')

                self.run_testplay(i)
                self.plot_winning_rate(i)
                self.hist_winning_rate(i)

        ray.shutdown()
        print('COMPLATED')


    def run_selfplay(self, j):
        self.network.to('cpu')
        start = time.time()
        work_in_progresses = [selfplay.remote(self.network) for i in range(20)]

        for i in trange(config.num_selfplay):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            self.replaybuffer.save_game(ray.get(finished)[0])
            work_in_progresses.extend([selfplay.remote(self.network)])

        print(f'{j}th run_selfplay:{time.time() - start}sec |', end='')

    def train_network(self, j):
        start = time.time()

        self.network.train()
        self.network.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

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

        values_ini, policies_ini, hidden_states_ini = self.network.initial_inference(encoded_states, device)
        values_rec, policies_rec = self.network.prediction(hidden_states)

        recon_states, mu, logvar = self.network.vae(encoded_states, device)


        v_loss = self.v_loss_fn(target_values, values_ini) + self.v_loss_fn(target_values, values_rec)
        p_loss = self.p_loss_fn(target_policies, policies_ini) + self.p_loss_fn(target_policies, policies_rec)
        vae_loss = loss_function(recon_states, encoded_states, mu, logvar)

        total_loss = (p_loss + v_loss + vae_loss)
        total_loss.backward()
        self.optimizer.step()

        self.total_losses.append(total_loss.detach().cpu().numpy())
        self.v_losses.append(v_loss.detach().cpu().numpy())
        self.p_losses.append(p_loss.detach().cpu().numpy())
        self.vae_losses.append(p_loss.detach().cpu().numpy())

    def run_testplay(self, j):

        work_in_progresses = [testplay.remote() for i in range(5)]
        result = []
        for i in range(config.num_testplay):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            result.append(ray.get(finished)[0])
            work_in_progresses.extend([testplay.remote()])

        result = np.array(result)
        zero_score = np.sum(result==0)
        one_score = np.sum(result==1)

        self.win_rate.append(one_score/config.num_testplay)

        if zero_score > one_score:
            print('newest model lose')
            checkpoint = torch.load('../var/champion.pth')
            self.network.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('newest model win')
            checkpoint = torch.load('../var/challenger.pth')
            self.network.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        torch.save(self.network.state_dict() , f'../models/model{j}.pth')

    def plot_each_loss(self):
        if len(self.total_losses) > 200:
            sampled_indices = np.linspace(0, len(self.total_losses) - 1, 200, dtype=int)
            sampled_indices = list(sampled_indices.astype(int))
            sampled_v_losses = [self.v_losses[i] for i in sampled_indices]
            sampled_p_losses = [self.p_losses[i] for i in sampled_indices]
            sampled_vae_losses = [self.vae_losses[i] for i in sampled_indices]
            plt.plot(range(len(sampled_v_losses)), sampled_v_losses, label='value_loss')
            plt.plot(range(len(sampled_p_losses)), sampled_p_losses, label='policy_loss')
            plt.plot(range(len(sampled_vae_losses)), sampled_vae_losses, label='vae_loss')

        else:
            plt.plot(range(len(self.v_losses)), self.v_losses, label='value_loss')
            plt.plot(range(len(self.p_losses)), self.p_losses, label='policy_loss')
            plt.plot(range(len(self.p_losses)), self.p_losses, label='vae_loss')

        plt.legend()
        plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
        plt.xlabel(f'epoch:{len(self.total_losses)}')
        plt.ylabel('Loss')
        plt.savefig(f'../out/each_losses/each_loss_{len(self.v_losses)}.pdf')
        plt.close()

    def plot_total_loss(self):
        if len(self.total_losses) > 200:
            sampled_indices = np.linspace(0, len(self.total_losses) - 1, 200, dtype=int)
            sampled_indices = list(sampled_indices.astype(int))
            sampled_total_losses = [self.total_losses[i] for i in sampled_indices]

            plt.plot(range(len(sampled_total_losses)), sampled_total_losses)
        else:
            plt.plot(range(len(self.total_losses)), self.total_losses)

        plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
        plt.xlabel(f'epoch:{len(self.total_losses)}')
        plt.ylabel('Loss')
        plt.savefig(f'../out/total_losses/learning_curve_{len(self.total_losses)}.pdf')
        plt.close()

    def plot_losses(self):
        plt.figure(figsize=(15, 3))
        if len(self.total_losses) > 200:
            sampled_indices = np.linspace(0, len(self.total_losses) - 1, 200, dtype=int)
            sampled_indices = list(sampled_indices.astype(int))
            sampled_v_losses = [self.v_losses[i] for i in sampled_indices]
            sampled_p_losses = [self.p_losses[i] for i in sampled_indices]
            sampled_vae_losses = [self.vae_losses[i] for i in sampled_indices]
            sampled_total_losses = [self.v_losses[i] for i in sampled_indices]

            plt.subplot(1, 4, 1)
            plt.plot(range(len(sampled_v_losses)), sampled_v_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('v_loss')

            plt.subplot(1, 4, 2)
            plt.plot(range(len(sampled_p_losses)), sampled_p_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('p_loss')

            plt.subplot(1, 4, 3)
            plt.plot(range(len(sampled_vae_losses)), sampled_vae_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('vae_loss')

            plt.subplot(1, 4, 4)
            plt.plot(range(len(sampled_total_losses)), sampled_total_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('total_loss')

        else:
            plt.subplot(1, 4, 1)
            plt.plot(range(len(self.v_losses)), self.v_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('v_loss')

            plt.subplot(1, 4, 2)
            plt.plot(range(len(self.p_losses)), self.p_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('p_loss')

            plt.subplot(1, 4, 3)
            plt.plot(range(len(self.vae_losses)), self.vae_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('vae_loss')

            plt.subplot(1, 4, 4)
            plt.plot(range(len(self.total_losses)), self.total_losses)
            plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
            plt.title('total_loss')

        plt.suptitle(f'epoch:{len(self.total_losses)}')
        plt.tight_layout()

        plt.savefig(f'../out/losses/learning_curve_{len(self.total_losses)}.pdf')
        plt.close()


    def plot_winning_rate(self, i):
        plt.scatter(range(len(self.win_rate)), self.win_rate)
        plt.xlabel('updates')
        plt.ylabel('winning rate')
        plt.title(f'{i}th iter', y=-0.25)
        plt.savefig(f'../out/winning_rates/winning_rate_{i}.pdf')
        plt.close()


    def hist_winning_rate(self):
        plt.figure(figsize=(3, 3))
        plt.hist(self.win_rate, bins=10, range=(0, 1), density=True, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='dashed', linewidth=2)

        plt.title('winnig ratio against past self')
        plt.tight_layout()
        plt.savefig('../out/winning_rates/winning_rate_against_past_self_{i}.pdf')
        plt.close()

@ray.remote
def selfplay(network):
    game = Game()
    while not game.is_terminated() and len(game.history['action']) < config.max_moves:
        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network.initial_inference(current_observation, device='cpu') #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)


    return game


@ray.remote
def testplay():
    network1 = Network()
    network2 = Network()

    a = random.randint(0, 1)
    if a == 0:
        network1.load_state_dict(torch.load('../var/champion.pth', map_location=torch.device('cpu'))['model'])
        network2.load_state_dict(torch.load('../var/challenger.pth', map_location=torch.device('cpu'))['model'])
    elif a == 1:
        network1.load_state_dict(torch.load('../var/challenger.pth', map_location=torch.device('cpu'))['model'])
        network2.load_state_dict(torch.load('../var/champion.pth', map_location=torch.device('cpu'))['model'])
    else:
        raise ValueError('Exception in flipping coin')

    game = Game()
    while not game.is_terminated() and len(game.history['action']) < config.max_moves:
        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network1.initial_inference(current_observation, device='cpu') #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network1)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)

        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network2.initial_inference(current_observation, device='cpu') #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network2)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)

    winner = game.environment.check_win()

    if a == 0 and winner == 1:
        return 0
    elif a == 0 and winner == -1:
        return 1
    elif a == 1 and winner == 1:
        return 1
    elif a == 1 and winner == -1:
        return 0

reconstruction_function = nn.MSELoss()
def loss_function(recon_x, x, mu, logvar):

    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD


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
            net_output = network.initial_inference(current_observation, device='cpu') #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)

    game.environment.print_board()
    game.environment.check_win2()
    target_value, target_policy = game.get_target(55)

    print(f'target_value:{target_value}')
    print(f'target_policy:{target_policy}')


if __name__ == '__main__':
    #_test()
    muzero = MuZero()
    muzero.run()