import pickle
import numpy as np

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""


class SVM:
    def __init__(self, path, learning_rate=0.0008, lambda_value=0.03, n_iters=1000):
        self.path = path
        self.lr = learning_rate
        self.lambda_value = lambda_value
        self.n_iters = n_iters
        self.block_size = self.__set_block_size()
        self.bounds_x = self.__set_bounds()[0]
        self.bounds_y = self.__set_bounds()[1]
        self.w = np.ndarray(shape=(4, 10), dtype=float)
        self.b = np.ndarray(shape=(4, 1))
        self.X = np.ndarray(shape=(3000, 10), dtype=int)
        self.y = np.ndarray(shape=(3000, 1), dtype=int)

    def read_from_file(self):
        with open(self.path, 'rb') as f:
            data_file = pickle.load(f)
        return data_file

    def __set_block_size(self):
        data = self.read_from_file()
        return data['block_size']

    def __set_bounds(self):
        data = self.read_from_file()
        return data['bounds']

    def fill_attributes(self):
        data = self.read_from_file()
        j = 1
        length_snake = 0
        for idx, move in enumerate(data["data"]):
            if length_snake > len(move[0]['snake_body']):
                j += 1
            if idx - j >= 3000:
                break
            head_x = move[0]['snake_body'][-1][0]
            head_y = move[0]['snake_body'][-1][1]
            direction = move[0]['snake_direction'].value
            distance_up = self.distance_up(head_x, head_y, move[0]['snake_body'])
            distance_right = self.distance_right(head_x, head_y, move[0]['snake_body'])
            distance_down = self.distance_down(head_x, head_y, move[0]['snake_body'])
            distance_left = self.distance_left(head_x, head_y, move[0]['snake_body'])
            apple_x = move[0]['food'][0]
            apple_y = move[0]['food'][1]
            length_snake = len(move[0]['snake_body'])

            self.X.itemset((idx - j, 7), head_x // self.block_size + 1)
            self.X.itemset((idx - j, 8), head_y // self.block_size + 1)
            self.X.itemset((idx - j, 2), direction)
            self.X.itemset((idx - j, 3), distance_up)
            self.X.itemset((idx - j, 4), distance_right)
            self.X.itemset((idx - j, 5), distance_down)
            self.X.itemset((idx - j, 6), distance_left)
            self.X.itemset((idx - j, 0), apple_x // self.block_size + 1)
            self.X.itemset((idx - j, 1), apple_y // self.block_size + 1)
            self.X.itemset((idx - j, 9), length_snake)
            self.y.itemset(idx - j, move[1].value)

    def distance_up(self, head_x, head_y, body):
        distance = head_y // self.block_size + 1
        for c in body:
            if head_x == c[0] and head_y > c[1]:
                distance = min(distance, (head_y - c[1]) // self.block_size)
        return distance

    def distance_right(self, head_x, head_y, body):
        distance = (self.bounds_x - head_x) // self.block_size
        for c in body:
            if head_y == c[1] and head_x < c[0]:
                distance = min(distance, (c[0] - head_x) // self.block_size)
        return distance

    def distance_down(self, head_x, head_y, body):
        distance = (self.bounds_y - head_y) // self.block_size
        for c in body:
            if head_x == c[0] and head_y < c[1]:
                distance = min(distance, (c[1] - head_y) // self.block_size)
        return distance

    def distance_left(self, head_x, head_y, body):
        distance = head_x // self.block_size + 1
        for c in body:
            if head_y == c[1] and head_x > c[0]:
                distance = min(distance, (head_x - c[0]) // self.block_size)
        return distance

    def game_state_to_attributes(self, move):
        list = np.ndarray(shape=(1, 10))
        head_x = move['snake_body'][-1][0]
        head_y = move['snake_body'][-1][1]
        direction = move['snake_direction'].value
        distance_up = self.distance_up(head_x, head_y, move['snake_body'])
        distance_right = self.distance_right(head_x, head_y, move['snake_body'])
        distance_down = self.distance_down(head_x, head_y, move['snake_body'])
        distance_left = self.distance_left(head_x, head_y, move['snake_body'])
        apple_x = move['food'][0]
        apple_y = move['food'][1]
        length_snake = len(move['snake_body'])
        list.itemset((0, 7), head_x // self.block_size + 1)
        list.itemset((0, 8), head_y // self.block_size + 1)
        list.itemset((0, 2), direction)
        list.itemset((0, 3), distance_up)
        list.itemset((0, 4), distance_right)
        list.itemset((0, 5), distance_down)
        list.itemset((0, 6), distance_left)
        list.itemset((0, 0), apple_x // self.block_size + 1)
        list.itemset((0, 1), apple_y // self.block_size + 1)
        list.itemset((0, 9), length_snake)
        return list

    def y_to_up_vs_rest(self):
        new_y = []
        for i in self.y:
            if i == 0:
                new_y.append(-1)
            else:
                new_y.append(1)
        return new_y

    def y_to_right_vs_rest(self):
        new_y = []
        for i in self.y:
            if i == 1:
                new_y.append(-1)
            else:
                new_y.append(1)
        return new_y

    def y_to_down_vs_rest(self):
        new_y = []
        for i in self.y:
            if i == 2:
                new_y.append(-1)
            else:
                new_y.append(1)
        return new_y

    def y_to_left_vs_rest(self):
        new_y = []
        for i in self.y:
            if i == 3:
                new_y.append(-1)
            else:
                new_y.append(1)
        return new_y

    def fit(self, y, direction):
        n_samples, n_attributes = self.X.shape
        w = np.zeros(n_attributes)
        b = 0
        y = y
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(self.X):
                condition = (y[idx] * (np.dot(x_i, w) - b) >= 1)
                if condition:
                    w -= self.lr * (2 * self.lambda_value * w)
                else:
                    w -= self.lr * (2 * self.lambda_value * w - np.dot(x_i, y[idx]))
                    b -= self.lr * y[idx]
        for idx in range(n_attributes):
            self.w.itemset((direction, idx), w[idx])
        self.b.itemset((direction, 0), b)

    def learn(self):
        self.fit(self.y_to_up_vs_rest(), 0)
        self.fit(self.y_to_right_vs_rest(), 1)
        self.fit(self.y_to_down_vs_rest(), 2)
        self.fit(self.y_to_left_vs_rest(), 3)

    def game_state_to_data_sample_class(self, attributes):
        up = np.dot(attributes, self.w[0]) - self.b[0]
        right = np.dot(attributes, self.w[1]) - self.b[1]
        down = np.dot(attributes, self.w[2]) - self.b[2]
        left = np.dot(attributes, self.w[3]) - self.b[3]
        direction = min(up, right, down, left)
        if direction == up:
            return 0
        elif direction == right:
            return 1
        elif direction == down:
            return 2
        elif direction == left:
            return 3

    def testing(self):
        good = 0
        y = self.y[2400:]
        for idx, move in enumerate(self.X[2400:]):
            if self.game_state_to_data_sample_class(move) == y[idx]:
                good += 1
        return good/600


if __name__ == "__main__":
    path1 = 'data/2022.11.29.22.31.51.pickle'
    example = SVM(path1)
    example.fill_attributes()
    example.learn()
    #print(example.w)
    #print(example.b)
    print(example.testing())
