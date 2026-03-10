import numpy as np

class RealGoGame:
	def __init__(self, board_size=3):
		self.board_size = board_size
		self.reset()

	def reset(self):
		"""
		Resets the board.
		Board convention: 1 = Black, -1 = White, 0 = Empty
		"""
		self.board = np.zeros((self.board_size, self.board_size), dtype=int)
		self.current_player = 1 
		self.passes = 0
		self.history = [] # Used for Ko checks
		self.captured_stones = {1: 0, -1: 0} # Track prisoners
		return self.board.copy()

	def get_valid_moves(self):
		"""
		Returns a list of all legal moves (r, c) + "PASS".
		Checks for:
		1. Occupied spots
		2. Suicide (unless it kills opponent)
		3. Ko (repeating board state)
		"""
		moves = []
		for r in range(self.board_size):
			for c in range(self.board_size):
				if self.board[r, c] == 0:
					# Test if this move is legal (Simulate it)
					if self._is_legal_move(r, c):
						moves.append((r, c))
		moves.append("PASS")
		return moves

	def step(self, action):
		"""
		Executes a move.
		Action: Tuple (r, c) or "PASS"
		"""
		if action == "PASS":
			self.passes += 1
			# Add current state to history for Ko checks
			self.history.append(self.board.copy())
			self.current_player *= -1
			return self.board.copy()
		
		self.passes = 0
		r, c = action
		
		# 1. Place Stone
		self.board[r, c] = self.current_player
		
		# 2. Check Opponent Captures (Remove dead opponent groups)
		opponent = -self.current_player
		captured_any = False
		
		# Check all adjacent neighbors for opponent groups
		neighbors = self._get_neighbors(r, c)
		for nr, nc in neighbors:
			if self.board[nr, nc] == opponent:
				group, liberties = self._get_group_and_liberties(nr, nc)
				if liberties == 0:
					self._remove_group(group)
					self.captured_stones[self.current_player] += len(group)
					captured_any = True

		# 3. Check Suicide (Own group has 0 liberties)
		# Note: In standard rules, if you capture, suicide is valid. 
		# If you didn't capture and have 0 liberties, it's illegal.
		# But `_is_legal_move` should have filtered purely illegal suicides.
		# This check is a sanity safeguard.
		group, liberties = self._get_group_and_liberties(r, c)
		if liberties == 0 and not captured_any:
			# Revert (Illegal Suicide) - Should not happen if get_valid_moves is used
			self.board[r, c] = 0 
			raise ValueError("Illegal Suicide Move attempted")

		# 4. Save State (for Ko) and Switch Turn
		self.history.append(self.board.copy())
		# Optimization: Keep history short (AlphaZero only needs last 8, but Ko needs full game theoretically. 
		# For 3x3, full game history is tiny, so we keep it all.)
		
		self.current_player *= -1
		return self.board.copy()

	def is_game_over(self):
		# Game ends on double pass or if board is full (optional safeguard)
		return self.passes >= 2
	
	def get_reward(self):
		"""
		Tromp-Taylor rules (Area Scoring):
		Score = (Stones on board) + (Empty points surrounded by stones)
		For 3x3, we can often just use stone count + prisoners as a proxy, 
		but still doing simple Area Scoring.
		"""
		black_area = 0
		white_area = 0
		
		# Naive ownership calculation (Flood fill empty spots)
		# If an empty region reaches ONLY Black stones, it's Black territory.
		visited = set()
		
		for r in range(self.board_size):
			for c in range(self.board_size):
				if self.board[r, c] == 1:
					black_area += 1
				elif self.board[r, c] == -1:
					white_area += 1
				elif (r, c) not in visited:
					# Empty spot, calculate territory
					group, owners = self._get_empty_region(r, c)
					visited.update(group)
					if 1 in owners and -1 not in owners:
						black_area += len(group)
					elif -1 in owners and 1 not in owners:
						white_area += len(group)
						
		# Komi is usually 7.5, but for 3x3 we might use 0 or smaller.
		# Let's use 0 for simplicity.
		if black_area > white_area: return 1
		if white_area > black_area: return -1
		return 0

	# --- Internal Helper Methods ---

	def _is_legal_move(self, r, c):
		# 1. Temporarily place stone
		self.board[r, c] = self.current_player
		
		# 2. Check for captures
		opponent = -self.current_player
		captures_made = False
		neighbors = self._get_neighbors(r, c)
		for nr, nc in neighbors:
			if self.board[nr, nc] == opponent:
				group, liberties = self._get_group_and_liberties(nr, nc)
				if liberties == 0:
					captures_made = True
					break # Valid move if it captures
					
		# 3. Check for suicide
		# If no captures, we must have liberties ourselves
		if not captures_made:
			group, liberties = self._get_group_and_liberties(r, c)
			if liberties == 0:
				self.board[r, c] = 0 # Undo
				return False

		# 4. Check Ko (Global state repetition)
		# We need to simulate the board *after* captures are removed
		temp_board = self.board.copy()
		if captures_made:
			self._remove_captured_groups_on_board(temp_board, r, c)
			
		# Check if this state exists in history
		# (Simple exact match check)
		is_ko = False
		for past_state in self.history:
			if np.array_equal(temp_board, past_state):
				is_ko = True
				break
				
		self.board[r, c] = 0 # Undo
		return not is_ko

	def _get_neighbors(self, r, c):
		neighbors = []
		for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
			nr, nc = r + dr, c + dc
			if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
				neighbors.append((nr, nc))
		return neighbors

	def _get_group_and_liberties(self, r, c):
		"""
		Flood fills to find the connected group of stones and its liberty count.
		"""
		color = self.board[r, c]
		group = set()
		liberties = set()
		stack = [(r, c)]
		
		while stack:
			curr = stack.pop()
			if curr in group: continue
			group.add(curr)
			
			for nr, nc in self._get_neighbors(curr[0], curr[1]):
				if self.board[nr, nc] == 0:
					liberties.add((nr, nc))
				elif self.board[nr, nc] == color and (nr, nc) not in group:
					stack.append((nr, nc))
					
		return group, len(liberties)

	def _remove_group(self, group):
		for r, c in group:
			self.board[r, c] = 0

	def _remove_captured_groups_on_board(self, board, r, c):
		"""
		Helper for Ko check to simulate captures on a temp board.
		Uses a localized flood-fill to check liberties against the temp board.
		"""
		color = board[r, c]
		opponent = -color
		
		for nr, nc in self._get_neighbors(r, c):
			if board[nr, nc] == opponent:
				# 1. Flood-fill to find the opponent group and its liberties on the TEMP board
				group = set()
				liberties = set()
				stack = [(nr, nc)]
				
				while stack:
					curr = stack.pop()
					if curr in group: continue
					group.add(curr)
					
					for nnr, nnc in self._get_neighbors(curr[0], curr[1]):
						if board[nnr, nnc] == 0:
							liberties.add((nnr, nnc))
						elif board[nnr, nnc] == opponent and (nnr, nnc) not in group:
							stack.append((nnr, nnc))
							
				# 2. If the group has exactly 0 liberties, remove it from the temp board
				if len(liberties) == 0:
					for gr, gc in group:
						board[gr, gc] = 0
				
	def _get_empty_region(self, r, c):
		"""Flood fill for scoring empty territories"""
		group = set()
		owners = set()
		stack = [(r, c)]
		while stack:
			curr = stack.pop()
			if curr in group: continue
			group.add(curr)
			
			for nr, nc in self._get_neighbors(curr[0], curr[1]):
				if self.board[nr, nc] == 0 and (nr, nc) not in group:
					stack.append((nr, nc))
				elif self.board[nr, nc] != 0:
					owners.add(self.board[nr, nc])
		return group, owners
