# reinforcement-learning-MAPD
The Multi-Agent Package Delivery (MAPD) problem with uncertainty solved using an implementation of the reinforcement learning algorithm Policy Iteration.

![image](https://github.com/user-attachments/assets/5e9d5de7-44c6-4441-962c-0766e9009b8b)

# Multi-Agent Package Delivery with Policy Iteration

This project implements a reinforcement learning approach using Policy Iteration to solve the Multi-Agent Package Delivery (MAPD) problem in a grid-based environment with uncertainty. The agents must navigate the grid while accounting for potential obstacles such as fragile edges that may become blocked during execution.

## Overview

The goal of the project is to enable an agent to traverse a grid environment and deliver a package from a start location to a goal location. The environment contains "fragile" edges that have a certain probability of being blocked, adding uncertainty to the agent's decision-making process. The agent makes decisions using the Policy Iteration algorithm, leveraging the Bellman Equation to find the optimal path to the goal while avoiding blocked paths.

## Policy Iteration Algorithm

Policy Iteration is an iterative algorithm used to compute the optimal policy and value function for a given Markov Decision Process (MDP). It alternates between two steps:

1. **Policy Evaluation**: For a fixed policy, compute the expected utility (value) of each state.
2. **Policy Improvement**: Update the policy by selecting the action that maximizes the expected utility for each state.

### Bellman Equation

The Bellman equation defines the value of a state as:

![image](https://github.com/user-attachments/assets/e188d3df-58b3-424b-927c-04c10422e182)


Where:
- \(V(s)\) is the value of state \(s\),
- \(R(s)\) is the immediate reward for being in state \(s\),
- \(\gamma\) is the discount factor,
- \(P(s'|s, a)\) is the transition probability from state \(s\) to state \(s'\) by taking action \(a\),
- \(V(s')\) is the value of the next state \(s'\).

In this project, Policy Iteration uses the Bellman Equation to iteratively refine the value function and update the policy accordingly.

## Input Explanation

The input to the program is a text file that defines the environment. The input file contains:
- Grid size (X, Y)
- Start position (S)
- Goal position (G)
- List of fragile edges with blocking probabilities (F)
- List of permanently blocked edges (B)

### Example Input

#X 2 <br />
#Y 1 <br />
#F 1 1 2 1 0.4 <br />
#F 2 0 2 1 0.2 <br />
#G 2 1 5 <br />
#S 0 0 <br />

## Usage Instructions

1. Clone the repository and navigate to the project directory.
2. Provide the input files in the specified format (`input1.txt`, `input2.txt`).
3. Run the Python script:

   ```ruby
   python reinforcement_learning_MAPD.py
   ```

4. Modify the input_num variable in the script to switch between different input files.
5. Follow the on-screen instructions to run multiple simulations with different instances of blocked fragile edges.

## License

This project is licensed under the MIT License.

## Requirements

This project uses Python 3.x and requires no external libraries beyond Python’s standard library.
