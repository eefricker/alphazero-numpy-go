import numpy as np
import datetime as dt
import random
from collections import deque

from environment import RealGoGame
from nn_numpy import (
    encode_input_tensor, conv_bn_relu_forward, residual_block_forward, 
    policy_head, value_head, value_head_backward, policy_head_backward, 
    residual_block_backward, conv_bn_relu_backward, init_random_params
)
from mcts import run_mcts

class AlphaZeroTrainer:
	def __init__(self, params, learning_rate=0.01, weight_decay=1e-4, sgd_momentum=0.9):
		self.params = params
		self.lr = learning_rate
		self.weight_decay = weight_decay
		self.sgd_momentum = sgd_momentum
		self.grads = {}
		self.velocities = {k: np.zeros_like(v) for k, v in self.params.items()}
		
	def train_step(self, x_batch, pi_targets, v_targets):
		N = x_batch.shape[0]
		caches = {}
		
		# --- 1. Forward Pass ---
		
		# A. Stem (Conv -> BN -> ReLU)
		stem_conv, caches['stem_conv'] = conv_bn_relu_forward(
			x_batch, 
			self.params['stem_w'], self.params['stem_b'],
			self.params['stem_bn_g'], self.params['stem_bn_b'],
			self.params['stem_bn_m'], self.params['stem_bn_v'],
			pad=1,mode='train'
		)
		
		# B. Residual Block
		res_params = {
			'w1': self.params['res_w1'], 'b1': self.params['res_b1'],
			'bn1_g': self.params['res_bn1_g'], 'bn1_b': self.params['res_bn1_b'],
			'bn1_m': self.params['res_bn1_m'], 'bn1_v': self.params['res_bn1_v'],
			'w2': self.params['res_w2'], 'b2': self.params['res_b2'],
			'bn2_g': self.params['res_bn2_g'], 'bn2_b': self.params['res_bn2_b'],
			'bn2_m': self.params['res_bn2_m'], 'bn2_v': self.params['res_bn2_v']
		}
		res_out, caches['res_block'] = residual_block_forward(stem_conv, res_params,mode='train')
		
		# --- C. Heads ---
		# 1. Policy Head
		p_probs, caches['p_head'] = policy_head(res_out, self.params, mode='train')
		
		# 2. Value Head
		v_pred, caches['v_head'] = value_head(res_out, self.params, mode='train')
		
		# --- 2. Loss Calculation ---
		d_v_logits = ((v_pred - v_targets) * (1 - v_pred**2) )
		d_p_logits = (p_probs - pi_targets)
		
		# --- 3. Backward Pass ---
		
		# A. Heads Backward
		d_res_from_v, v_grads = value_head_backward(d_v_logits, caches['v_head'], self.params)
		self.grads.update(v_grads)

		d_res_from_p, p_grads = policy_head_backward(d_p_logits, caches['p_head'], self.params)
		self.grads.update(p_grads)

		# Combine Head Gradients at Residual Output
		d_res_out = d_res_from_v + d_res_from_p
		
		# B. Residual Block Backward
		d_stem_out, res_grads = residual_block_backward(d_res_out, caches['res_block'], res_params)
		# res_grads uses keys like 'w1', 'bn1_g', so we prefix them
		for k, v in res_grads.items():
			self.grads[f'res_{k}'] = v
			
		# C. Stem Backward
		dx, stem_grads = conv_bn_relu_backward(d_stem_out, caches['stem_conv'], 
											   self.params['stem_w'], self.params['stem_b'], pad=1)
		for k, v in stem_grads.items():
			self.grads[f'stem_{k}'] = v
			
		# --- 4. SGD Update ---
		l2_penalty_loss = 0.0
		effective_lr = self.lr / N  # Average the step size here instead
		
		for key in self.params:
			if key in self.grads:
				
				# Clipping
				self.grads[key] = np.clip(self.grads[key], -1.0, 1.0)
				
				# 1. Apply L2 Regularization (only to weights)
				if key.endswith('_w'):
					self.grads[key] += self.weight_decay * self.params[key]
					l2_penalty_loss += 0.5 * self.weight_decay * np.sum(self.params[key]**2)
				
				# 2. Update Velocity: v = (momentum * v) + (lr * grad)
				self.velocities[key] = (self.sgd_momentum * self.velocities[key]) + (effective_lr * self.grads[key])
				
				# 3. Apply Velocity to Parameters: w = w - v
				self.params[key] -= self.velocities[key]
				
		# Return losses
		l_v = np.mean((v_pred - v_targets)**2)
		l_p = -np.mean(np.sum(pi_targets * np.log(p_probs + 1e-7), axis=1))
		return l_v, l_p, l2_penalty_loss
		
def get_symmetries(board_size, state, pi, z):
	"""
	Generates all 8 symmetries for a given state and policy.
	state: (17, board_size, board_size)
	pi: 1D array of length (board_size^2 + 1)
	z: scalar value
	"""
	symmetries = []
	
	# 1. Separate the board policy from the PASS policy
	pi_board = pi[:-1].reshape(board_size, board_size)
	pi_pass = pi[-1]

	for i in range(4):
		# A. Rotations (0, 90, 180, 270 degrees)
		# For state (C, H, W), we rotate along spatial axes 1 and 2
		rot_state = np.rot90(state, k=i, axes=(1, 2))
		rot_pi_board = np.rot90(pi_board, k=i)
		
		# Flatten and append the PASS move back
		rot_pi = np.append(rot_pi_board.flatten(), pi_pass)
		symmetries.append((rot_state.copy(), rot_pi.copy(), z))
		
		# B. Reflections (Horizontal Flip of the rotated state)
		# Flip along axis 2 (Width) for the state tensor
		flip_state = np.flip(rot_state, axis=2)
		# fliplr works perfectly for the 2D policy matrix
		flip_pi_board = np.fliplr(rot_pi_board)
		
		flip_pi = np.append(flip_pi_board.flatten(), pi_pass)
		symmetries.append((flip_state.copy(), flip_pi.copy(), z))
		
	return symmetries
	
def execute_self_play(params, num_games=1):
	dataset = []
	
	for g in range(num_games):
		game = RealGoGame(board_size=3)
		game_history = [] 
		moves_count = 0
		
		# Initialize an empty root for the start of the game
		mcts_root = None 
		
		while not game.is_game_over() and moves_count < 40: 
			# Pass the current tree root in, and catch the updated tree
			pi, mcts_root = run_mcts(game, params, current_root=mcts_root, num_simulations=100)
			
			if moves_count < 6:
				action_idx = np.random.choice(len(pi), p=pi)
			else:
				action_idx = np.argmax(pi)
			
			if action_idx == 3*3:
				move = "PASS"
			else:
				move = (action_idx // 3, action_idx % 3)
				
			input_state = encode_input_tensor(game)
			game_history.append([input_state, pi, game.current_player])
			
			# Step the actual game board forward
			game.step(move)
			moves_count += 1
			
			# STEP THE MCTS TREE FORWARD
			# Make the chosen child node the new root.
			# Python's garbage collector will automatically delete the unchosen branches.
			if move in mcts_root.children:
				mcts_root = mcts_root.children[move]
			else:
				# Fallback safeguard (should not happen in self-play unless valid_moves logic breaks)
				mcts_root = None

		winner = game.get_reward()
		
		for state, pi, player in game_history:
			
			z = winner * player
			
			# Generate the 8 symmetric variations of this single turn
			sym_data = get_symmetries(game.board_size, state, pi, z)
			
			# Add all 8 to our dataset
			dataset.extend(sym_data)
			
	return dataset

class ReplayBuffer:
	def __init__(self, max_size=10000):
		# deque automatically drops the oldest items when max_size is reached
		self.buffer = deque(maxlen=max_size)
		
	def add(self, game_data):
		"""
		game_data is a list of tuples: [(state, pi, z), (state, pi, z), ...]
		generated from a single self-play game.
		"""
		self.buffer.extend(game_data)
		
	def sample(self, batch_size):
		"""
		Returns a random mini-batch of (state, pi, z) tuples.
		"""
		return random.sample(self.buffer, batch_size)
		
	def __len__(self):
		return len(self.buffer)

def complete_training_loop(training_games, training_steps_per_game=10, min_buffer_size=1000):
	tic = dt.datetime.today()
	params = init_random_params(board_size=3)
	trainer = AlphaZeroTrainer(params, learning_rate=0.01)

	# For 3x3, a max size of 10,000 is great. 
	# It holds roughly the last 500-600 games.
	replay_buffer = ReplayBuffer(max_size=10000)
	batch_size = 64 

	print("Global Parameters Initialized.")
	print(f"Starting Training Loop for {training_games} Training Games...")
	print(f"Burn-in phase: waiting for {min_buffer_size} states before training.\n")

	for i in range(training_games):
		
		# A. Self-Play: Generate Data and ADD to Buffer
		raw_data = execute_self_play(params, num_games=1)
		replay_buffer.add(raw_data)
		
		# B. Do not train until we cross the threshold
		if len(replay_buffer) < min_buffer_size:
			# Print a status update during burn-in so you know it isn't frozen
			if (i + 1) % 5 == 0:
				print(f"Burn-in progress: {len(replay_buffer)}/{min_buffer_size} states collected...")
			continue
			
		# C. Gradient Descent: Update 'params' MULTIPLE times per game generated
		total_loss_v, total_loss_p, total_loss_l2 = 0, 0, 0
		
		for _ in range(training_steps_per_game):
			# Because the buffer is now large, this random sample will have 
			# highly de-correlated, independent states.
			batch = replay_buffer.sample(batch_size)
			states, policies, values = zip(*batch)
			
			x_batch = np.array(states)               
			pi_batch = np.array(policies)            
			v_batch = np.array(values).reshape(-1, 1)
			
			loss_v, loss_p, loss_l2 = trainer.train_step(x_batch, pi_batch, v_batch)
			
			total_loss_v += loss_v
			total_loss_p += loss_p
			total_loss_l2 += loss_l2
			
		# Average the losses for reporting
		avg_loss_v = total_loss_v / training_steps_per_game
		avg_loss_p = total_loss_p / training_steps_per_game
		avg_loss_l2 = total_loss_l2 / training_steps_per_game
		
		time_elapsed = dt.datetime.today() - tic
		s_per_game = time_elapsed.seconds / (i + 1)
		
		if (i % 250 == 249):
			print(f"=== Completed {i+1} iterations, Time elapsed = {time_elapsed}, s/game = {s_per_game:.2f} ===")
			print(f"  Buffer Size: {len(replay_buffer)}")
			print(f"  Avg Loss Value (MSE): {avg_loss_v:.4f}")
			print(f"  Avg Loss Policy (CE): {avg_loss_p:.4f}")
			print(f"  Avg Loss L2 Penalty: {avg_loss_l2:.4f}")
			print(f"  Total Loss: {avg_loss_v + avg_loss_p + avg_loss_l2:.4f}")
			print("-" * 30)

	print("\nTraining Complete")
	return params
