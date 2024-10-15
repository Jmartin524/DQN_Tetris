# DQN_Tetris
A Simple example of a neural network designed to learn to play Tetris


Learning Process in DQN
Experience Replay:

During each episode, the agent interacts with the Tetris environment by taking actions, observing the results, and receiving rewards.
Each experience (state, action, reward, next state, done) is stored in the agent's memory (a deque).
This memory allows the agent to sample random experiences during training, which helps in breaking the correlation between consecutive experiences.
Updating the Model:

After each action, the agent receives feedback in the form of a reward.
When the agent completes a step in the environment, it uses the experience to update its neural network.
The target Q-value is computed based on the reward received and the maximum Q-value of the next state. This is the key part of the learning process.
The model is then updated using the Mean Squared Error (MSE) loss between the predicted Q-values and the target Q-values.
Epsilon-Greedy Strategy:

The agent starts with a high exploration rate (epsilon), meaning it will choose random actions to explore the environment.
Over time, epsilon is gradually decreased (using epsilon_decay), which encourages the agent to exploit its learned knowledge rather than exploring.
Learning Across Episodes:

Each episode provides the agent with new experiences, helping it improve its policy for playing Tetris.
The model is saved after training (as shown in the previous code), which allows the agent to retain the knowledge it has gained across runs.
Example of How Learning Happens
In Episode 1: The agent might make a series of random moves since it starts with a high epsilon. It receives rewards and stores experiences.
In Episode 2: With a bit of learning from Episode 1, the agent will make better moves, potentially getting higher rewards. The replay memory will help it refine its policy based on previous experiences.
Subsequent Episodes: As the agent continues to play, it will learn to recognize patterns in Tetris, improving its performance and maximizing its score.
