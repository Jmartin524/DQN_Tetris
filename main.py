import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os  # Importing os

# Pygame setup
pygame.init()

# Game constants
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Tetris shapes (each represented as a list of coordinates)
SHAPES = [
    [[1, 1, 1], [0, 1, 0]],  # T-shape
    [[1, 1], [1, 1]],        # O-shape
    [[1, 1, 1, 1]],          # I-shape
    [[0, 1, 1], [1, 1, 0]],  # S-shape
    [[1, 1, 0], [0, 1, 1]]   # Z-shape
]

# Neural Network for DQN
class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        self.fc1 = nn.Linear(GRID_WIDTH * GRID_HEIGHT + 2, 128)  # Include piece position in state
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)  # Left, Right, Rotate, Drop

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = TetrisNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            loss = F.mse_loss(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Tetris Game class
class Tetris:
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.current_piece = self.new_piece()
        self.piece_x, self.piece_y = 4, 0
        self.is_game_over = False
        self.score = 0

    def new_piece(self):
        return random.choice(SHAPES)

    def check_collision(self, piece, offset_x, offset_y):
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell and (
                    x + self.piece_x + offset_x >= GRID_WIDTH or 
                    x + self.piece_x + offset_x < 0 or
                    y + self.piece_y + offset_y >= GRID_HEIGHT or
                    self.grid[y + self.piece_y + offset_y][x + self.piece_x + offset_x]):
                    return True
        return False

    def rotate_piece(self):
        rotated = [list(row) for row in zip(*self.current_piece[::-1])]
        if not self.check_collision(rotated, 0, 0):
            self.current_piece = rotated

    def drop(self):
        if not self.check_collision(self.current_piece, 0, 1):
            self.piece_y += 1
        else:
            self.lock_piece()
            self.clear_lines()
            self.current_piece = self.new_piece()
            self.piece_x, self.piece_y = 4, 0
            if self.check_collision(self.current_piece, 0, 0):
                self.is_game_over = True

    def move(self, dx):
        if not self.check_collision(self.current_piece, dx, 0):
            self.piece_x += dx

    def lock_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[y + self.piece_y][x + self.piece_x] = 1

    def clear_lines(self):
        full_lines = [i for i, row in enumerate(self.grid) if all(row)]
        for i in full_lines:
            self.grid = np.delete(self.grid, i, axis=0)
            self.grid = np.vstack([[0] * GRID_WIDTH, self.grid])
            self.score += 1

    def draw_grid(self, screen):  # Accept screen as an argument
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 0:
                    pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)
                else:
                    pygame.draw.rect(screen, GREEN, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def draw_piece(self, screen):  # Accept screen as an argument
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, RED, ((x + self.piece_x) * BLOCK_SIZE, (y + self.piece_y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def get_state(self):
        state = self.grid.flatten().tolist() + [self.piece_x, self.piece_y]
        return np.array(state, dtype=np.float32)

    def update(self):
        self.drop()

# Main game loop
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Tetris with DQN')
    agent = DQNAgent(state_size=GRID_WIDTH * GRID_HEIGHT + 2, action_size=4)  # Include position in state
    clock = pygame.time.Clock()
    EPISODES = 1000  # Number of episodes to train
    BATCH_SIZE = 32  # Batch size for training

    for episode in range(EPISODES):
        game = Tetris()  # Reset the game for each episode
        state = game.get_state()
        total_reward = 0

        while not game.is_game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            action = agent.act(state)  # Choose action based on current state
            
            # Map action to game function
            if action == 0:  # Move left
                game.move(-1)
            elif action == 1:  # Move right
                game.move(1)
            elif action == 2:  # Rotate
                game.rotate_piece()
            elif action == 3:  # Drop
                game.drop()
            
            # Get the next state and reward
            next_state = game.get_state()
            if game.is_game_over:
                reward = -1  # Negative reward for game over
            else:
                reward = 1  # Positive reward for continuing the game
            
            # Store the experience in the agent's memory
            agent.remember(state, action, reward, next_state, game.is_game_over)

            # Train the agent
            agent.replay(BATCH_SIZE)

            # Update the current state
            state = next_state
            total_reward += reward

            # Pygame update and drawing
            screen.fill(WHITE)
            game.draw_grid(screen)  # Pass the screen to draw_grid
            game.draw_piece(screen)  # Pass the screen to draw_piece
            game.update()
            pygame.display.update()
            clock.tick(10)  # Limit to 10 FPS
        
        print(f"Episode: {episode + 1}, Score: {game.score}, Total Reward: {total_reward}")

    print("Training finished.")

if __name__ == "__main__":
    main()
