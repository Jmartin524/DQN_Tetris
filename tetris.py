import pygame
import random
import numpy as np

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

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Tetris')

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

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == 0:
                    pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)
                else:
                    pygame.draw.rect(screen, GREEN, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def draw_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, RED, ((x + self.piece_x) * BLOCK_SIZE, (y + self.piece_y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def update(self):
        self.drop()

# Main game loop
def main():
    game = Tetris()
    clock = pygame.time.Clock()

    while not game.is_game_over:
        screen.fill(WHITE)
        game.draw_grid()
        game.draw_piece()
        game.update()

        pygame.display.update()
        clock.tick(10)  # Limit to 10 FPS

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move(-1)
                elif event.key == pygame.K_RIGHT:
                    game.move(1)
                elif event.key == pygame.K_DOWN:
                    game.drop()
                elif event.key == pygame.K_UP:
                    game.rotate_piece()

    print(f"Game Over! Your score: {game.score}")

if __name__ == "__main__":
    main()
