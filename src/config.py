# MuZero
num_iters = 10000


# selfplay
num_selfplay = 50 #60 #20

# training
num_epoch = 150  #200 #5
num_unroll_steps = 1
batch_size = 16
checkpoint_interval = 1

# game
board_length = 8
max_moves = 60
state_history_len = 1
state_shape = (state_history_len*2+1, 8, 8)

# network
action_space_size = 65
num_channels = 128
num_hidden = 8

# replaybuffer
window_size = 100

# MCTS
num_simulations = 200 #200 #10
discount = 1

# UCB formula
pb_c_base = 19652
pb_c_init = 1.25

# test play
test_interval = 3 #
num_testplay = 15