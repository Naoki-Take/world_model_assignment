import config

class RepalyBuffer:
    def __init__(self):
        self.window_size = config.window_size
        self.buffer_size = 0
        self.buffer = []

    def save_game(self, game):

        if len(self.buffer) == self.window_size:
            del self.buffer[0]
            self.buffer_size -= 1
        self.buffer.append(game)
        self.buffer_size += 1
    
    def get_len(self):
        return len(self.buffer)