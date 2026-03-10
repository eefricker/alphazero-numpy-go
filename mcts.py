import numpy as np
import copy
from nn_numpy import run_neural_net

class MCTS_Node:
	def __init__(self, prior):
		self.visit_count = 0
		self.value_sum = 0
		self.prior = prior
		self.children = {}
		self.state = None 

	def value(self):
		return self.value_sum / (self.visit_count + 1e-6)

def run_mcts(root_game, params, current_root=None, num_simulations=100, dirichlet_alpha=1.0, dirichlet_epsilon=0.25):
	# 1. Use existing root, or create a new one if starting fresh
	if current_root is None:
		root = MCTS_Node(prior=0)
	else:
		root = current_root
		
	# 2. Ensure the root is expanded (it might already be expanded from previous turns)
	if len(root.children) == 0:
		policy, _ = run_neural_net(root_game, params)
		valid_moves = root_game.get_valid_moves()
		
		valid_probs = []
		for move in valid_moves:
			if move == "PASS": idx = root_game.board_size**2
			else: idx = move[0] * root_game.board_size + move[1]
			valid_probs.append(policy[idx])
		
		policy_sum = sum(valid_probs)
		if policy_sum > 1e-6:
			valid_probs = [p / policy_sum for p in valid_probs]
		else:
			valid_probs = [1.0 / len(valid_moves)] * len(valid_moves)
			
		for i, move in enumerate(valid_moves):
			root.children[move] = MCTS_Node(prior=valid_probs[i])

	# 3. Temporarily apply Dirichlet Noise to the root's children
	valid_moves = list(root.children.keys())
	noise = np.random.dirichlet([dirichlet_alpha] * len(valid_moves))
	
	original_priors = {}
	for i, move in enumerate(valid_moves):
		child = root.children[move]
		original_priors[move] = child.prior # Save original NN prior
		# Apply formula: (1 - eps) * P + eps * noise
		child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i]
		
	# --- EXISTING: Main Simulation Loop ---
	for _ in range(num_simulations):
		node = root
		scratch_game = copy.deepcopy(root_game) 
		search_path = [node]
		
		# A. Selection (Your existing UCB logic goes here)
		while len(node.children) > 0:
			best_score = -float('inf')
			best_action = -1
			best_child = None
			
			for action, child in node.children.items():
				# UCB Formula
				ucb = child.value() + 1.4 * child.prior * np.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
				
				if np.isnan(ucb):
					ucb = -float('inf')

				if ucb > best_score:
					best_score = ucb
					best_action = action
					best_child = child
			
			if best_action == -1:
				best_action, best_child = list(node.children.items())[0]

			node = best_child
			scratch_game.step(best_action)
			search_path.append(node)
			
			if scratch_game.is_game_over():
				break

		# B. Evaluation & C. Expansion (For all non-root nodes)
		value = 0
		if scratch_game.is_game_over():
			winner = scratch_game.get_reward()
			value = winner * scratch_game.current_player
		else:
			policy, value = run_neural_net(scratch_game, params)
			
			valid_moves = scratch_game.get_valid_moves()
			policy_sum = 0
			
			for move in valid_moves:
				if move == "PASS":
					idx = scratch_game.board_size**2
				else:
					idx = move[0] * scratch_game.board_size + move[1]
				
				prob = policy[idx]
				node.children[move] = MCTS_Node(prior=prob)
				policy_sum += prob
			
			if policy_sum > 1e-6:
				for child in node.children.values():
					child.prior /= policy_sum
			else:
				n_children = len(node.children)
				for child in node.children.values():
					child.prior = 1.0 / n_children

		# D. Backup (Your existing backup logic goes here)
		for node in reversed(search_path):
			value = -value
			node.value_sum += value
			node.visit_count += 1
			
	# 4. Restore the original priors after search is done
	for move, child in root.children.items():
		child.prior = original_priors[move]
		
	# ... (Your existing code to return the 'pi' array goes here)
	board_size = root_game.board_size
	pi = np.zeros(board_size * board_size + 1)
	
	for action, child in root.children.items():
		if action == "PASS":
			idx = board_size**2
		else:
			idx = action[0] * board_size + action[1]
		pi[idx] = child.visit_count
		
	pi_sum = np.sum(pi)
	if pi_sum > 0:
		pi /= pi_sum
	else:
		pi[:] = 1.0 / len(pi)
		
	return pi, root
