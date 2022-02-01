import re
import torch
import numpy as np
import random
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
from argparse import ArgumentParser, RawDescriptionHelpFormatter

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.003
INPUT_SIZE = 12
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 3
EPSILON = 10
GAMMA = 0.95


class Agent:
    def __init__(self, epsilon=EPSILON, gamma=GAMMA, load_model=None) -> None:
        self.n_games = 0
        self.epsilon = epsilon  # randomness
        self.gamma = gamma  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        if load_model is not None:
            self.model.load(load_model)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        points_snake_r = [Point(i[0] + 1, i[1]) for i in game.snake]
        points_snake_u = [Point(i[0], i[1] + 1) for i in game.snake]
        points_snake_l = [Point(i[0] - 1, i[1]) for i in game.snake]
        points_snake_d = [Point(i[0], i[1] - 1) for i in game.snake]

        state = [
            # danger ahead
            (dir_r and game._is_collision(point_r))
            or (dir_l and game._is_collision(point_l))
            or (dir_u and game._is_collision(point_u))
            or (dir_d and game._is_collision(point_d)),
            # danger right
            (dir_u and game._is_collision(point_r))
            or (dir_d and game._is_collision(point_l))
            or (dir_l and game._is_collision(point_u))
            or (dir_r and game._is_collision(point_d)),
            # danger left
            (dir_d and game._is_collision(point_r))
            or (dir_u and game._is_collision(point_l))
            or (dir_r and game._is_collision(point_u))
            or (dir_l and game._is_collision(point_d)),
            # # snake ahead
            # (dir_l in points_snake_r)
            # or (dir_r in points_snake_l)
            # or (dir_d in points_snake_u)
            # or (dir_u in points_snake_d),
            # # snake right
            # (dir_d in points_snake_r)
            # or (dir_u in points_snake_l)
            # or (dir_r in points_snake_u)
            # or (dir_l in points_snake_d),
            # # snake left
            # (dir_u in points_snake_r)
            # or (dir_d in points_snake_l)
            # or (dir_l in points_snake_u)
            # or (dir_r in points_snake_d),
            # 2nd last move
            # (dir_l in game.snake[-2])
            # or (dir_r in game.snake[-2])
            # or (dir_d in game.snake[-2])
            # or (dir_u in game.snake[-2]),
            # 3rd last move
            (dir_l in game.snake[-3])
            or (dir_r in game.snake[-3])
            or (dir_d in game.snake[-3])
            or (dir_u in game.snake[-3]),
            # move direction,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves. Exploration vs Exploitation
        if self.n_games > 250:
            self.epsilon = 0
        elif self.n_games > 150:
            self.epsilon = 1
        # self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            _state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(_state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def play(load_model=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(load_model=load_model)
    agent.model.eval()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        _, done, score = game.play_step(final_move)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            if score > record:
                record = score

            print(f"Game {agent.n_games} Score {score} Record {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, title="Playing")


def train(load_model=None, savepath=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(load_model=load_model)
    agent.model.train()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save(savepath)

            print(f"Game {agent.n_games} Score {score} Record {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    usage = "%prog [options] file (or - for stdin)\n"
    parser = ArgumentParser(
        usage,
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Example of usage:
    """,
    )
    parser.add_argument(
        "-m", "--model", action="store", type=str, dest="model", default=None
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store",
        type=str,
        dest="save",
        default="./model/model.pth",
    )
    parser.add_argument(
        "-p",
        "--play",
        action="store_true",
    )

    args = parser.parse_args()
    savepath = args.save

    if args.play:
        print("play only")
        play(load_model=args.model)
    else:
        print(f"saving model to {savepath}")
        train(load_model=args.model, savepath=savepath)
