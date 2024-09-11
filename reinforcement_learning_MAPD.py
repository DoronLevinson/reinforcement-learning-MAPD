import os
import copy
import random

''' PART 1: Reading the input '''

def read_input(input_path):
    """
    Reads input from a file and parses the grid size, start/goal positions,
    blocked edges, fragile edges, and time limit.
    
    Args:
    - input_path (str): Path to the input file
    
    Returns:
    - x, y (int): Grid dimensions (x_max, y_max)
    - start, goal (tuple): Start and goal positions
    - blocked_edges (list of tuples): List of permanently blocked edges
    - fragile_edges (list of Fragile_edge): List of fragile edges with probabilities
    - time_limit (int): Maximum time limit for the agent to reach the goal
    """
    with open(input_path, 'r') as file:
        lines = file.readlines()

    fragile_edges = []
    blocked_edges = []

    for line in lines:
        if line.startswith('#X'):
            x = int(line.split()[1])
        elif line.startswith('#Y'):
            y = int(line.split()[1])
        elif line.startswith('#F'):
            elements = line.split()
            v1 = (int(elements[1]), int(elements[2]))
            v2 = (int(elements[3]), int(elements[4]))
            prob = float(elements[5])
            fragile_edges.append(Fragile_edge(v1, v2, prob))
        elif line.startswith('#B'):
            # Blocked edges can be traversed both ways, so we add both directions.
            edge_coords = list(map(int, line.split()[1:]))
            blocked_edge = ((edge_coords[0], edge_coords[1]), (edge_coords[2], edge_coords[3]))
            blocked_edge_reversed = ((edge_coords[2], edge_coords[3]), (edge_coords[0], edge_coords[1]))
            blocked_edges.extend([blocked_edge, blocked_edge_reversed])
        elif line.startswith('#S'):
            start = (int(line.split()[1]), int(line.split()[2]))
        elif line.startswith('#G'):
            goal = (int(line.split()[1]), int(line.split()[2]))
            time_limit = int(line.split()[3])

    return x, y, start, goal, blocked_edges, fragile_edges, time_limit


class Fragile_edge:
    """
    Represents a fragile edge with a certain probability of being blocked.
    
    Attributes:
    - v1, v2 (tuple): Coordinates of the vertices of the edge.
    - prob (float): Probability that the edge will be blocked.
    - values (list): Values indicating whether the edge is intact (0), blocked (1), or has a probability (prob).
    """
    def __init__(self, v1, v2, prob):
        self.v1 = v1
        self.v2 = v2
        self.prob = prob
        self.values = [0, prob, 1]  # Possible states of the edge


''' PART 2: Creating the Value Function '''

def initialize_belief_states(x_max, y_max, fragile_edges, goal):
    """
    Initializes the belief states (possible states in the environment) and the value function.
    
    Args:
    - x_max, y_max (int): Grid dimensions.
    - fragile_edges (list): List of fragile edges.
    - goal (tuple): Goal state.
    
    Returns:
    - value_f (dict): Value function initialized with rewards at goal states and penalties at time limit.
    """
    belief_states = []

    for x in range(x_max + 1):
        for y in range(y_max + 1):
            for t in range(time_limit + 1):
                belief_states.append((x, y, t))
    
    belief_states = add_fragile(fragile_edges, belief_states)
    value_f = {}
    
    for item in belief_states:
        if (item[0], item[1]) == goal:
            value_f[item] = 1  # Reward for reaching the goal
        elif item[2] == time_limit:
            value_f[item] = -1  # Penalty for exceeding time limit
        else:
            value_f[item] = 0  # Initial neutral value
    
    value_f = irregular(value_f)  # Mark unreachable states as irregular
    return value_f

def add_fragile(fragile_edges, belief_states=[()], i=0):
    """
    Adds possible states for fragile edges to the belief states.
    
    Args:
    - fragile_edges (list): List of fragile edges.
    - belief_states (list): List of current belief states.
    - i (int): Index of the current fragile edge being processed.
    
    Returns:
    - belief_states (list): Updated belief states with fragile edges added.
    """
    if i == len(fragile_edges):
        return belief_states
    
    new_belief_states = []
    for state in belief_states:
        state = list(state)
        for value in fragile_edges[i].values:
            new_state = state.copy()
            new_state.append(value)
            new_belief_states.append(tuple(new_state))
    
    return add_fragile(fragile_edges, new_belief_states, i + 1)

def irregular(value_f):
    """
    Marks unreachable states (irregular) based on the state of fragile edges.
    
    Args:
    - value_f (dict): The current value function.
    
    Returns:
    - value_f (dict): Updated value function with irregular states marked.
    """
    for s, u in value_f.items():
        for i, f in enumerate(fragile_edges):
            if (f.v1 == (s[0], s[1]) or f.v2 == (s[0], s[1])) and (s[i + 3] != 0 and s[i + 3] != 1):
                value_f[s] = "irregular"  # Mark as irregular if the edge state is not clear
    
    return value_f


''' PART 3: Value Iteration Algorithm '''

def value_iteration(value_f, policy):
    """
    Performs the value iteration algorithm to update the value function and policy.
    
    Args:
    - value_f (dict): Current value function.
    - policy (dict): Current policy.
    
    Returns:
    - value_f (dict): Updated value function.
    - policy (dict): Updated policy with optimal actions.
    """
    for s, u in value_f.items():
        if u == "irregular" or (s[0], s[1]) == goal or s[2] == time_limit:
            continue  # Skip goal, time limit, and irregular states
        
        cur_blocked_edges = add_blocked(s)
        location = (s[0], s[1])

        # Determine possible actions (right, left, up, down, stay)
        right = (min(x, s[0] + 1), s[1])
        if (location, right) in cur_blocked_edges:
            right = location
        left = (max(0, s[0] - 1), s[1])
        if (location, left) in cur_blocked_edges:
            left = location
        up = (s[0], min(y, s[1] + 1))
        if (location, up) in cur_blocked_edges:
            up = location
        down = (s[0], max(0, s[1] - 1))
        if (location, down) in cur_blocked_edges:
            down = location

        actions = [right, left, up, down, location]
        action_names = ['right', 'left', 'up', 'down', 'X']  # 'X' means stay in place

        max_utility = float('-inf')
        for i, new_location in enumerate(actions):
            # Determine relevant fragile edges revealed in this move
            relevant_edges = find_relevant_edges(s, new_location)
            
            # Compute new states and their probabilities after moving
            new_state = list(s)
            new_state[0] = new_location[0]
            new_state[1] = new_location[1]
            new_state[2] = s[2] + 1
            new_states = create_new_states(relevant_edges, new_state)

            # Calculate expected utility of the new states
            utility = 0
            for state in new_states:
                utility += value_f[state[0]] * state[1]
            
            if utility >= max_utility:
                max_utility = utility
                max_action_name = action_names[i]
                max_location = new_location

        # Update the value of the state s using the Bellman equation
        value_f[s] = round(step_cost + gamma * max_utility, 2)
        policy[s] = (max_action_name, max_location)
    
    return value_f, policy

def find_relevant_edges(s, new_location):
    """
    Finds fragile edges that will be revealed based on the agent's new location.
    
    Args:
    - s (tuple): Current state.
    - new_location (tuple): Agent's new location.
    
    Returns:
    - relevant_edges (list): List of fragile edges that are revealed in this move.
    """
    relevant_edges = []
    for i, fragile_edge in enumerate(fragile_edges):
        if (new_location == fragile_edge.v1 or new_location == fragile_edge.v2) and (s[i + 3] != 1 and s[i + 3] != 0):
            relevant_edges.append((i + 3, fragile_edge.prob))  # Append edge index and probability
    
    return relevant_edges

def create_new_states(relevant_edges, state, i=0, prob=1):
    """
    Recursively creates new states based on the outcomes of fragile edges.
    
    Args:
    - relevant_edges (list): List of fragile edges revealed in the move.
    - state (list): Current state.
    - i (int): Index of the fragile edge being processed.
    - prob (float): Probability of the current state.
    
    Returns:
    - list of tuples: List of possible new states and their probabilities.
    """
    if i == len(relevant_edges):
        return [(tuple(state), prob)]
    
    # Update the state based on whether the fragile edge is blocked or not
    state[relevant_edges[i][0]] = 1  # Edge blocked
    res1 = create_new_states(relevant_edges, state, i + 1, prob * relevant_edges[i][1])

    state[relevant_edges[i][0]] = 0  # Edge intact
    res2 = create_new_states(relevant_edges, state, i + 1, prob * (1 - relevant_edges[i][1]))

    return res1 + res2

def add_blocked(state):
    """
    Adds blocked edges based on the current state of fragile edges.
    
    Args:
    - state (tuple): Current state of the agent.
    
    Returns:
    - new_blocked_edges (list): Updated list of blocked edges.
    """
    new_blocked_edges = blocked_edges.copy()
    for i in range(len(state)):
        if i >= 3 and state[i] == 1:
            v1 = fragile_edges[i - 3].v1
            v2 = fragile_edges[i - 3].v2
            new_blocked_edges.extend([(v1, v2), (v2, v1)])  # Add both directions of the blocked edge
    
    return new_blocked_edges


'''PART 4: Simulation'''

def run_simulation(instance=[]):
    """
    Runs a simulation of the agent's path based on the current policy and a given instance of fragile edge outcomes.
    
    Args:
    - instance (list): List of fragile edge outcomes (0: intact, 1: blocked). If not provided, a random instance is generated.
    """
    if len(instance) == 0:
        instance = [random.randint(0, 1) for _ in fragile_edges]

    state = [start[0], start[1], 0]
    for i, edge in enumerate(fragile_edges):
        if start == edge.v1 or start == edge.v2:
            state.append(instance[i])
        else:
            state.append(edge.prob)

    state = tuple(state)
    path = [start]
    
    while state in policy:
        path.extend([policy[state][0], policy[state][1]])

        new_state = [policy[state][1][0], policy[state][1][1], state[2] + 1]
        for i, edge in enumerate(fragile_edges):
            if policy[state][1] == edge.v1 or policy[state][1] == edge.v2:
                new_state.append(instance[i])
            else:
                new_state.append(state[i + 3])
        state = tuple(new_state)
    
    print(f"Simulation Instance: {instance}")
    print(f"Path: {path}\n")


''' PART 5: Main '''

# Reading input and initializing the environment
input_num = 1  # Change input_num to 2 for another scenario
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, f'input{input_num}.txt')  # Adjust file path if needed
x, y, start, goal, blocked_edges, fragile_edges, time_limit = read_input(file_path)

# Global constants
iterations = 5
reward = 1
step_cost = -0.01
gamma = 1

# Calculate the policy and value function using Value Iteration
value_f = initialize_belief_states(x, y, fragile_edges, goal)
policy = {}

for _ in range(iterations):
    value_f, policy = value_iteration(value_f, policy)

print(f"\nValue Function:\n{value_f}\n\nPolicy:\n{policy}\n")

# Example simulation
if input_num == 1:
    run_simulation([0, 1])
    run_simulation([1, 1])
if input_num == 2:
    run_simulation([0])
    run_simulation([1])

# Optional random simulations
while True:
    key = input("Press Enter to run one more simulation, press E to exit... \n")
    if key.lower() == 'e':
        exit(0)
    else:
        run_simulation()