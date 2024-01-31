from network1 import *
from tqdm import trange
import numpy as np
import ray
from game import *
import torch
import random
from mcts import *

def run_testplay():
    win_rate = []
    work_in_progresses = [testplay.remote() for i in range(20)]
    result = []
    for i in trange(100):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        result.append(ray.get(finished)[0])
        work_in_progresses.extend([testplay.remote()])

    result = np.array(result)
    zero_score = np.sum(result==0)
    one_score = np.sum(result==1)

    print(one_score/100)



def testplay():

    a = random.randint(0, 1)
    if a == 0:
        network1 = Network1()
        network2 = Network()
        network1.load_state_dict(torch.load('./var/muzero.pth', map_location=torch.device('cpu')))
        network2.load_state_dict(torch.load('./var/muzero_vae.pth', map_location=torch.device('cpu')))
    elif a == 1:
        network1 = Network()
        network2 = Network1()
        network1.load_state_dict(torch.load('./var/muzero_vae.pth', map_location=torch.device('cpu')))
        network2.load_state_dict(torch.load('./var/muzero.pth', map_location=torch.device('cpu')))
    else:
        raise ValueError('Exception in flipping coin')

    game = Game()
    state = initial_state()
    print_board(state, game.environment.steps)
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
        print_board(game.environment.observe(), game.environment.steps)

        if game.is_terminated() or len(game.history['action']) > config.max_moves:
            break

        root = Node(prior=0)
        current_observation = game.get_encoded_state(game.environment.steps)
        current_observation = torch.from_numpy(current_observation)
        with torch.no_grad():
            net_output = network2.initial_inference(current_observation, device='cpu') #! on cpu
        root.expand_node(game.legal_actions(), net_output)

        action = MCTS(root, game, network2)
        game.apply(Action(action, game.to_play()))
        game.store_search_statistics(root)
        print_board(game.environment.observe(), game.environment.steps)

    winner = game.environment.check_win()

    if a == 0 and winner == 1:
        return 0
    elif a == 0 and winner == -1:
        return 1
    elif a == 1 and winner == 1:
        return 1
    elif a == 1 and winner == -1:
        return 0


def print_board(state, i):

    board = state[0] - state[1]

    char_board = np.vectorize(index_to_char)(board)
    with open(f'../out/board/turn{i}.txt', 'w') as f:
        f.write("  0  1  2  3  4  5  6  7\n")
        for i in range(8):
            f.write(f'{i}')
            for j in range(8):
                f.write(f' {char_board[i][j]} ')
            f.write('\n')


def index_to_char(index):
    if index == 1:
        return '●'
    elif index == 0:
        return '-'
    elif index == -1:
        return '○'
    else:
        print(index)
        raise Exception("これは例外です")


def initial_state():
    state = np.zeros((2, 8, 8), dtype=np.int8)
    state[0, 3, 4] = 1
    state[0, 4, 3] = 1
    state[1, 3, 3] = 1
    state[1, 4, 4] = 1

    return state


import os
import config
import torch
import numpy as np
from network import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.stats import multivariate_normal

import shutil

def empty_folder(folder_path):
    try:
        # フォルダ内のファイルとサブフォルダを削除
        shutil.rmtree(folder_path)
        # 空のフォルダを再作成
        os.makedirs(folder_path)
        print(f"フォルダ '{folder_path}' の中身を空にしました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

###############################################################################################################
## check reconstruction ratio
def draw_reconstruction_ratio(model_folder_path):

    reconstruction_ratios = []
    file_list = os.listdir(model_folder_path)

    for file_name in file_list:
        file_path = os.path.join(model_folder_path, file_name)
        reconstruction_succeeded = 0

        sampled_states = sample_encoded_state(file_path, n=100)
        for sampled_state in sampled_states:
            assert sampled_state.shape == (2, 8, 8) , f'sampled_state.shape:{sampled_state.shape}'

            reconstruction_succeeded += check_board(sampled_state)

        reconstruction_ratios.append(reconstruction_succeeded/100)

    plt.plot(range(len(reconstruction_ratios)), reconstruction_ratios)
    plt.xlabel('updates')
    plt.ylabel('winning rate')
    plt.savefig(f'../out/reconstruction_rates/reconstruction_rates.pdf')
    plt.close()


def sample_encoded_state(file_path, n=1, threshold = 0.5):
    sampled_vector = torch.randn((n,128))
    #sampled_tensor = torch.tensor(sampled_vector).view(n, config.num_hidden, config.board_length, config.board_length)

    network = Network()
    network.state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    sampled_state = network.vae.decode(sampled_vector)[:, :2, :, :] # third layer means nothing here

    sampled_state = torch.where(sampled_state > threshold, torch.tensor(1.0), torch.tensor(0.0)).numpy()

    return sampled_state

def check_board(state):

    board = state[0] + state[1]
    board = np.clip(board, None, 1.0)

    res = 0

    for i in range(config.board_length):
        for j in range(config.board_length):
            if board[i][j] == 1:
                board = dfs(board, i, j)
                res += 1

    if res == 1:
        return 1
    else:
        return 0

def dfs(board, x, y):

    board[x][y] = 0

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx = x + dx
            ny = y + dy

            if (0 <= nx < config.board_length) and (0 <= ny < config.board_length) and (board[nx][ny] == 1):
                dfs(board, nx, ny)

    return board

def test_dfs():
    state = initial_state()
    state[1][5][5] = 1
    return check_board(state)
###############################################################################################################

###############################################################################################################
## check latent space of vae
def z2image(z, threshold = 0.5): # z = (128, )

    h = torch.tensor(z).view(n, config.num_hidden, config.board_length, config.board_length)

    network = Network()
    network.state_dict(torch.load('../var/champion.pth', map_location=torch.device('cpu'))['model'])
    continuous_state = network.vae.decode(h)[0:2] # third layer means nothing here
    state = torch.where(continuous_state > threshold, torch.tensor(1.0), torch.tensor(0.0)).numpy()
    print_board(state)


def print_board(state, r, k):
    print(f'r:{r}')
    os.makedirs(f'../out/sample_latent_space/{r}', exist_ok=True)

    board = state[0] - state[1]

    char_board = np.vectorize(index_to_char)(board)
    with open(f'../out/sample_latent_space/{r}/file{k}.txt', 'w') as f:
        f.write("  0  1  2  3  4  5  6  7\n")
        for i in range(8):
            f.write(f'{i}')
            for j in range(8):
                f.write(f' {char_board[i][j]} ')
            f.write('\n')


def index_to_char(index):
    if index == 1:
        return '●'
    elif index == 0:
        return '-'
    elif index == -1:
        return '○'

###############################################################################################################

###############################################################################################################
## draw move in 2d normal
def sample_boards():
    a = True
    q = 0
    l = 0
    n = 0
    while a:
        l = sample_board(1000, 128, n, q)

        if l != 0:
            print(f'l:{l}')
            n += 1
            print(f'n:{n}')

        if n > 20:
            a = False

        l = 0
        q += 1

def sample_board(n, d, r, iter, file_path='../var/champion.pth'):
    #empty_folder(folder_path='../out/sample_latent_space')
    zs = sample_n_latents(n, d)
    l = zs2image(zs, r, iter,file_path)
    return l


def sample_n_latents(n, d):
    sampled_vector1 = 10000*torch.randn(d)
    sampled_vector2 = 10000*torch.randn(d)

    interpolation_weights = torch.linspace(0, 1, n)
    interpolated_vectors = sampled_vector1 + interpolation_weights[:, None] * (sampled_vector2 - sampled_vector1)

    linear_tensors = [tensor.view(1, -1) for tensor in interpolated_vectors]

    return linear_tensors


def zs2image(zs, r, iter, file_path):
    network = Network()
    network.state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    l = 0
    for i, z in enumerate(zs):
        sampled_state = network.vae.decode(z).squeeze()[:2, :, :] # third layer means nothing here
        sampled_state = torch.where(sampled_state > 0.5, torch.tensor(1.0), torch.tensor(0.0)).numpy()
        #print(f'sampled_state:{sampled_state[0], sampled_state[1]}')
        #print(f'multiply:{np.multiply(sampled_state[0], sampled_state[1])}')
        sum_of_elements = np.sum(np.multiply(sampled_state[0], sampled_state[1]))
        sum_0 = np.sum(sampled_state[0])
        sum_1 = np.sum(sampled_state[1])
        #print(f'sum_of_elements:{sum_of_elements}')
        if (sum_of_elements == 0) and (sum_0 >= 5) and (sum_1 >= 5) and check_board(sampled_state):
            print(f'r-:{r}')
            print_board(sampled_state, r, i)
            l+=1

    print(f'{iter}:success: {l}')
    return l

def draw_move_in_2D():

    #sampled_encoded_states =

    mean = [0, 0]
    covariance = [[1, 0], [0, 1]]

    # 2次元正規分布のオブジェクトを作成
    rv = multivariate_normal(mean, covariance)

    # メッシュグリッドの作成
    x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    pos = np.dstack((x, y))

    # 密度関数の計算
    z = rv.pdf(pos)

    # 等高線の描画
    plt.contour(x, y, z, levels=10, cmap='viridis')
    plt.title('2D Normal Distribution Contour')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    def draw_legal_reconstruction_rate():
        pass


def initial_state():
    state = np.zeros((2, 8, 8), dtype=np.int8)
    state[0, 3, 4] = 1
    state[0, 4, 3] = 1
    state[1, 3, 3] = 1
    state[1, 4, 4] = 1

    return state

    # a = [0.6, 0.66, 0.6, 0.46, 0.46, 0.66, 0.13, 0.4, 0.33, 0.46, 0.4, 0.8, 0.4, 0.4, 0.66, 0.53, 0.53, 0.6, 0.46, 0.6, 0.53, 0.4, 0.46, 0.33, 0.53, 0.46, 0.6]
    # b = [0.6, 0.73, 0.4, 0.53, 0.73, 0.4,0.67, 0.6, 0.2, 0.6, 0.8, 0.67, 0.6, 0.67, 0.4, 0.8, 0.53, 0.46, 0.6, 0.4, 0.6, 0.53, 0.67, 0.67, 0.46, 0.53, 0.46, 0.6]


    # plt.figure(figsize=(6, 3))

    # plt.subplot(1, 2, 1)
    # plt.hist(a, bins=10, range=(0, 1), density=True, color='skyblue', edgecolor='black')
    # plt.axvline(x=0.5, color='red', linestyle='dashed', linewidth=2)

    # #plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
    # plt.title('MuZero')

    # plt.subplot(1, 2, 2)
    # plt.hist(b, bins=10, range=(0, 1), density=True, color='skyblue', edgecolor='black')
    # plt.axvline(x=0.5, color='red', linestyle='dashed', linewidth=2)

    # #plt.gca().get_xaxis().set_major_locator(FixedLocator([]))
    # plt.title('MuZero-VAE')

    # plt.suptitle('winnig ratio against past self')
    # plt.tight_layout()
    # plt.savefig('../out/winning_rates/winning_rate_against_past_self.pdf')
    # plt.close()

if __name__ == '__main__':
    testplay()