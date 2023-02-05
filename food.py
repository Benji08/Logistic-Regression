import random


class Food:
    def __init__(self, block_size, bounds):
        self.color = (150, 0, 0)
        self.x, self.y = 0, 0
        self.block_size = block_size
        self.bounds = bounds

    def draw(self, game, window):
        game.draw.rect(window, self.color, (self.x, self.y, self.block_size, self.block_size))

    def respawn(self):
        blocks_in_x = (self.bounds[0]) / self.block_size
        blocks_in_y = (self.bounds[1]) / self.block_size
        self.x = random.randint(0, blocks_in_x - 1) * self.block_size
        self.y = random.randint(0, blocks_in_y - 1) * self.block_size
