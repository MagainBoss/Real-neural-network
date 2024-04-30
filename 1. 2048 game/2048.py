import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_episodes = 100000
size = 512


class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        self.score = 0

    def add_random_tile(self):
        empty_cells = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.board[i][j] == 0
        ]
        if empty_cells:
            i, j = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[i][j] = np.random.choice([2, 4], p=[0.9, 0.1])

    def move(self, direction):
        moved = False
        if direction == "up":
            for j in range(self.size):
                for i in range(1, self.size):
                    if self.board[i][j] != 0:
                        for k in range(i, 0, -1):
                            if self.board[k - 1][j] == 0:
                                self.board[k - 1][j] = self.board[k][j]
                                self.board[k][j] = 0
                                moved = True
                            elif self.board[k - 1][j] == self.board[k][j]:
                                self.board[k - 1][j] *= 2
                                self.board[k][j] = 0
                                self.score += self.board[k - 1][j]
                                moved = True
                                break
                            else:
                                break
        elif direction == "down":
            for j in range(self.size):
                for i in range(self.size - 2, -1, -1):
                    if self.board[i][j] != 0:
                        for k in range(i, self.size - 1):
                            if self.board[k + 1][j] == 0:
                                self.board[k + 1][j] = self.board[k][j]
                                self.board[k][j] = 0
                                moved = True
                            elif self.board[k + 1][j] == self.board[k][j]:
                                self.board[k + 1][j] *= 2
                                self.board[k][j] = 0
                                self.score += self.board[k + 1][j]
                                moved = True
                                break
                            else:
                                break
        elif direction == "left":
            for i in range(self.size):
                for j in range(1, self.size):
                    if self.board[i][j] != 0:
                        for k in range(j, 0, -1):
                            if self.board[i][k - 1] == 0:
                                self.board[i][k - 1] = self.board[i][k]
                                self.board[i][k] = 0
                                moved = True
                            elif self.board[i][k - 1] == self.board[i][k]:
                                self.board[i][k - 1] *= 2
                                self.board[i][k] = 0
                                self.score += self.board[i][k - 1]
                                moved = True
                                break
                            else:
                                break
        elif direction == "right":
            for i in range(self.size):
                for j in range(self.size - 2, -1, -1):
                    if self.board[i][j] != 0:
                        for k in range(j, self.size - 1):
                            if self.board[i][k + 1] == 0:
                                self.board[i][k + 1] = self.board[i][k]
                                self.board[i][k] = 0
                                moved = True
                            elif self.board[i][k + 1] == self.board[i][k]:
                                self.board[i][k + 1] *= 2
                                self.board[i][k] = 0
                                self.score += self.board[i][k + 1]
                                moved = True
                                break
                            else:
                                break
        return moved

    def step(self, action):
        moved = False
        if action == 0:
            moved = self.move("up")
        elif action == 1:
            moved = self.move("down")
        elif action == 2:
            moved = self.move("left")
        elif action == 3:
            moved = self.move("right")
        self.add_random_tile()
        return self.get_state(), self.get_score(), self.is_game_over(), moved

    def get_state(self):
        return self.board.copy()

    def get_score(self):
        return self.score

    def is_game_over(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return False

        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False

        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i][j] == self.board[i + 1][j]:
                    return False

        print(self.board)
        return True


class DQN2048(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN2048, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * input_shape[1] * input_shape[2], size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, num_actions)
        self.conv1.to(device)
        self.conv2.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)

    def forward(self, x):
        x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self, model, target_model, replay_buffer, gamma=0.99, batch_size=1, lr=1e-3
    ):
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon=0.1, num_actions=4):
        if np.random.rand() < epsilon:
            return np.random.randint(0, num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (
            1 - dones
        ) * self.gamma * next_q_values.unsqueeze(1)

        loss = torch.nn.functional.mse_loss(q_values, expected_q_values).to(device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


env = Game2048()

input_shape = (1, 4, 4)
num_actions = 4
model = DQN2048(input_shape, num_actions)
# model.load_state_dict(torch.load("dqn_model.pth"))
target_model = DQN2048(input_shape, num_actions)

model.to(device)
target_model.to(device)

replay_buffer = ReplayBuffer(capacity=10000)

agent = DQNAgent(model, target_model, replay_buffer)

if_train = True
if if_train:
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()
        done = False
        total_reward = 0
        max_reward = 0
        while not done:
            action = agent.select_action(state, num_actions=num_actions)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()
            agent.update_target_model()
            state = next_state
            total_reward += reward
            max_reward = max(max_reward, reward)
        print(
            f"Episode: {episode + 1}, Total Reward: {total_reward}, Max Reward: {max_reward}"
        )
        if episode % 100 == 0:
            torch.save(agent.model.state_dict(), "dqn_model.pth")

    torch.save(agent.model.state_dict(), "dqn_model.pth")


model = DQN2048(input_shape, num_actions)
model.load_state_dict(torch.load("dqn_model.pth"))
model.to(device)
model.eval()

agent = DQNAgent(model, model, replay_buffer)

test_episodes = 100
total_rewards = []

for episode in range(test_episodes):
    env.reset()
    state = env.get_state()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state, epsilon=0.01, num_actions=num_actions)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    total_rewards.append(total_reward)

average_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")
