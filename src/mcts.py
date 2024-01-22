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

