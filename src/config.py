# MuZero
num_iters = 1 #10000
test_interval = 10

# selfplay
num_selfplay = 100

# training

# game
board_length = 8
max_moves = 60
state_history_len = 1
state_shape = (state_history_len*2+1, 8, 8)

# network
action_space_size = 60
num_channels = 128
num_hidden = 8
