import config
from network import *
from game import *


def MCTS(root, game, network):
    player = game.to_play()
    for _ in range(config.num_simulations):
        node = root

        # choose node based on score if it already exists in tree
        while node.is_expanded():
            action, node = node.select_child()
            player = get_opponent_player(player)

        encoded_action = Action(action, get_opponent_player(player)).get_encoded_action()

        with torch.no_grad():
            network_output = network.recurrent_inference(node.parent.hidden_state, torch.from_numpy(encoded_action)) #! on cpu

        node.expand_node([Action(i, player) for i in range(config.action_space_size)], network_output)

        node.backpropagate(network_output[0].item())

    visit_counts = [(child.visit_count, action) for action, child in root.children.items()]

    if len(game.history['action']) < 5:
        t = 1.0
    else:
        t = 0.0  # Play according to the max.

    action = softmax_sample(visit_counts, t)
    
    return action


def softmax_sample(distribution, temperature: float):
    visits = [i[0] for i in distribution]
    actions = [i[1] for i in distribution]
    if temperature == 0:
        return actions[visits.index(max(visits))]
    elif temperature == 1:
        visits_sum = sum(visits)
        visits_prob = [i/visits_sum for i in visits]
        return np.random.choice(actions, p=visits_prob)
    else:
        raise NotImplementedError


def test_PvC():

    network = Network()

    print('black or white?')
    
    while True:
        color =  int(input('1.black 2.white'))
        if color in (1, 2):
            break
        else:
            print('input 1 or 2.')

    game = Game()
    game.environment.print_board()

    if color == 1:
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
            
            root = Node(prior=0)
            current_observation = game.get_encoded_state(game.environment.steps)
            current_observation = torch.from_numpy(current_observation)
            with torch.no_grad():
                net_output = network.initial_inference(current_observation) #! on cpu
            root.expand_node(game.legal_actions(), net_output)
            
            action = MCTS(root, game, network)

            game.apply(Action(action, game.to_play()))

            print(f'Cpu:{Action(action, game.to_play()).position}')
            game.environment.print_board()

            if game.is_terminated():
                game.environment.check_win2()
                break
    
    else:
        while True:
            
            root = Node(prior=0)
            current_observation = game.get_encoded_state(game.environment.steps)
            current_observation = torch.from_numpy(current_observation)
            with torch.no_grad():
                net_output = network.initial_inference(current_observation) #! on cpu
            root.expand_node(game.legal_actions(), net_output)
            
            action = MCTS(root, game, network)

            game.apply(Action(action, game.to_play()))
            print(f'Cpu:{Action(action, game.to_play()).position}')
            game.environment.print_board()

            if game.is_terminated():
                game.environment.check_win2()
                break
            
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

if __name__ == '__main__':
    test_PvC()