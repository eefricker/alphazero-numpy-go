import numpy as np

# Constants for the 3x3 scale
BOARD_SIZE = 3
HISTORY_LENGTH = 8
INPUT_PLANES = (HISTORY_LENGTH * 2) + 1  # 17 planes

def encode_input_tensor(game):
	input_tensor = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
	current_player = game.current_player
	opponent = -current_player
	
	# Construct sequence: [Current Board, t-1, t-2, ...]
	boards = [game.board.copy()] + list(reversed(game.history[:-1]))
	
	for i in range(HISTORY_LENGTH):
		if i < len(boards):
			state = boards[i]
			input_tensor[i] = (state == current_player).astype(np.float32)
			input_tensor[i + HISTORY_LENGTH] = (state == opponent).astype(np.float32)
	
	# Color plane
	if current_player == 1:
		input_tensor[16] = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
		
	return input_tensor

def relu(x):
	"""
	Rectified Linear Unit: f(x) = max(0, x)
	"""
	return np.maximum(0, x)

def dense_forward(x, w, b):
	"""
	Standard Fully Connected (Linear) Layer.
	
	Inputs:
	- x: Input data (N, D_in) - flattened input
	- w: Weights (D_in, D_out)
	- b: Biases (D_out,)
	"""
	# Matrix multiplication: (N, D_in) x (D_in, D_out) -> (N, D_out)
	return np.dot(x, w) + b

def softmax(x):
	"""
	Stable Softmax function.
	Subtracts max(x) to prevent numerical overflow with large exponents.

	Idea is to convert numbers into probability distributions
	"""
	# Shift x for stability
	# keepdims=True ensures we can broadcast the subtraction
	shifted_x = x - np.max(x, axis=1, keepdims=True)
	
	exp_x = np.exp(shifted_x)
	sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
	
	return exp_x / sum_exp_x

def tanh(x):
	"""Hyperbolic Tangent: Squeezes output between -1 and 1."""
	return np.tanh(x)

def conv2d_forward_slow(x, w, b, stride=1, pad=1):
	"""
	A naive implementation of a convolutional layer.
	
	Inputs:
	- x: Input data of shape (N, C_in, H, W)
	- w: Filter weights of shape (F, C_in, HH, WW)
	- b: Biases, of shape (F,)
	- stride: The number of pixels between adjacent receptive fields (int)
	- pad: The number of pixels that will be used to zero-pad the input (int)
	
	Returns:
	- out: Output data, of shape (N, F, H', W')

	Rough Idea:
	Takes given filters (board patterns), and integrates to see how much each part of the history boards matches up with the filter.
	Includes bias terms for each filter, and pads the history board which impacts dimensions.
	Example, a flag shape of stones could be a filter, and then you overlap it with a board from the history and dot product to get the
	similarity. Stride = 1 means check every possible overlap, increasing stride means skipping some overlaps
	
	"""
	N, C_in, H, W = x.shape
	F, _, HH, WW = w.shape
	
	# Calculate Output Dimensions
	# Formula: H' = 1 + (H + 2*pad - HH) / stride
	H_out = 1 + (H + 2 * pad - HH) // stride
	W_out = 1 + (W + 2 * pad - WW) // stride
	
	# Initialize Output
	out = np.zeros((N, F, H_out, W_out))
	
	# 1. Zero-pad the input
	# Pad only height and width dimensions: ((0,0), (0,0), (pad,pad), (pad,pad))
	x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
	
	# 2. The Sliding Window Loop
	# We iterate over every sample, every filter, and every spatial position
	
	for n in range(N):                  # For each image in batch
		for f in range(F):              # For each filter (kernel)
			for i in range(H_out):      # For each vertical position
				for j in range(W_out):  # For each horizontal position
					
					# Determine the corners of the current slice on the padded input
					vert_start = i * stride
					vert_end   = vert_start + HH
					horiz_start= j * stride
					horiz_end  = horiz_start + WW
					
					# Extract the slice (Receptive Field)
					# Shape: (C_in, HH, WW)
					x_slice = x_pad[n, :, vert_start:vert_end, horiz_start:horiz_end]
					
					# The Convolution Operation: Element-wise multiply and sum
					# We sum over all input channels (C_in), Height(HH), and Width(WW)
					# w[f] is the f-th filter with shape (C_in, HH, WW)
					out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
					
	return out
	
def conv2d_forward(x, w, b, stride=1, pad=1):
	"""
	Optimized Convolutional Layer using np.tensordot.
	Reduces Python loop iterations from N*F*H*W to just H*W.
	"""
	N, C_in, H, W = x.shape
	F, _, HH, WW = w.shape
	
	H_out = 1 + (H + 2 * pad - HH) // stride
	W_out = 1 + (W + 2 * pad - WW) // stride
	
	out = np.zeros((N, F, H_out, W_out))
	
	# 1. Zero-pad the input
	x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
	
	# 2. Loop ONLY over the 3x3 spatial output dimensions (9 iterations total)
	for i in range(H_out):
		for j in range(W_out):
			
			vert_start = i * stride
			vert_end   = vert_start + HH
			horiz_start= j * stride
			horiz_end  = horiz_start + WW
			
			# Extract the slice for the ENTIRE batch at once
			# Shape: (N, C_in, HH, WW)
			x_slice = x_pad[:, :, vert_start:vert_end, horiz_start:horiz_end]
			
			# Tensordot translates the multi-dimensional multiply-and-sum into optimized C math.
			# We want to multiply and sum over C_in, HH, and WW.
			# In x_slice, these are axes 1, 2, and 3.
			# In w, these are axes 1, 2, and 3.
			# The remaining axes (N from x_slice, F from w) become the output dimensions (N, F).
			out[:, :, i, j] = np.tensordot(x_slice, w, axes=([1, 2, 3], [1, 2, 3]))
			
	# Add bias to all filters (broadcasting across N, H, W automatically)
	# b shape is (F,), we reshape to (1, F, 1, 1) to add it cleanly
	out += b.reshape(1, F, 1, 1)
				
	return out

def batch_norm_forward(x, gamma, beta, mean, var, epsilon=1e-3, mode='train',momentum=0.9):
	
	N, C, H, W = x.shape
	if mode == 'train':
		# 1. Calculate actual batch statistics
		batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
		batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
		
		mean[:] = momentum * mean + (1.0 - momentum) * batch_mean.reshape(-1)
		var[:] = momentum * var + (1.0 - momentum) * batch_var.reshape(-1)
		
		x_minus_mean = x - batch_mean
		var_plus_eps = batch_var + epsilon
		std_inv = 1.0 / np.sqrt(var_plus_eps)
		x_norm = x_minus_mean * std_inv
		
		# 2. Scale and shift
		gamma_reshaped = gamma.reshape(1, C, 1, 1)
		beta_reshaped = beta.reshape(1, C, 1, 1)
		out = gamma_reshaped * x_norm + beta_reshaped
		
		# 3. Cache the calculated batch stats for the backward pass!
		cache = (x_norm, gamma, x_minus_mean, batch_var, epsilon)
		return out, cache
		
	else:
		mean_reshaped = mean.reshape(1, C, 1, 1)
		var_reshaped = var.reshape(1, C, 1, 1)
		gamma_reshaped = gamma.reshape(1, C, 1, 1)
		beta_reshaped = beta.reshape(1, C, 1, 1)
		
		# Normalize
		x_minus_mean = x - mean_reshaped
		var_plus_eps = var_reshaped + epsilon
		std_inv = 1.0 / np.sqrt(var_plus_eps)
		x_norm = x_minus_mean * std_inv
		
		# Scale & Shift
		out = gamma_reshaped * x_norm + beta_reshaped
		
		# Cache for backward
		cache = (x_norm, gamma, x_minus_mean, var_reshaped, epsilon)
		return out, cache

def conv_bn_relu_forward(x, w, b, bn_g, bn_b, bn_m, bn_v, pad, mode='inference'):
	out = conv2d_forward(x, w, b, pad=pad)
	out, bn_cache = batch_norm_forward(out, bn_g, bn_b, bn_m, bn_v,mode=mode)
	relu_in = out
	out = relu(out)
	return out, {'x': x, 'bn_cache': bn_cache, 'relu_in': relu_in}


def residual_block_forward(x, params, mode='inference'):
	"""
	Simplified Residual Block using the conv_bn_relu_forward helper.
	"""
	cache = {}
	cache['x'] = x # Save input for the skip connection
	
	# --- Block Part 1: Use the Helper ---
	# This replaces Conv1, BN1, and ReLU1
	r1_out, helper_cache = conv_bn_relu_forward(
		x, 
		params['w1'], params['b1'],
		params['bn1_g'], params['bn1_b'], 
		params['bn1_m'], params['bn1_v'],
		pad=1, mode=mode
	)
	
	# Unpack helper cache for backward compatibility
	# The existing backward pass expects specific keys:
	cache['bn1_cache'] = helper_cache['bn_cache']
	cache['bn1_out'] = helper_cache['relu_in'] # The input to ReLU is the output of BN
	cache['r1_out'] = r1_out
	
	# --- Block Part 2: Manual Implementation ---
	# We must do this manually because the ReLU comes AFTER the skip connection
	
	# 4. Conv2
	c2_out = conv2d_forward(r1_out, params['w2'], params['b2'], stride=1, pad=1)
	
	# 5. BN2
	# Note: We capture bn2_cache directly here
	bn2_out, cache['bn2_cache'] = batch_norm_forward(
		c2_out, params['bn2_g'], params['bn2_b'], 
		params['bn2_m'], params['bn2_v'], mode=mode
	)
	
	# 6. Add (Skip Connection)
	add_out = bn2_out + x
	cache['add_out'] = add_out
	
	# 7. Final ReLU
	final_out = relu(add_out)
	
	return final_out, cache

def policy_head(x, params, mode='inference'):
	cache = {}
	
	# 1. Conv -> BN -> ReLU (using helper)
	# Note: pad=0 is crucial here for 1x1 convolutions
	h, stem_cache = conv_bn_relu_forward(
		x, 
		params['p_w1'], params['p_b1'],
		params['p_bn_g'], params['p_bn_b'], 
		params['p_bn_m'], params['p_bn_v'],
		pad=0,mode=mode
	)
	
	# 2. Flatten
	N, C, H, W = h.shape
	h_flat = h.reshape(N, -1)
	
	# 3. Dense -> Softmax
	logits = dense_forward(h_flat, params['p_fc_w'], params['p_fc_b'])
	probs = softmax(logits)
	
	if mode == 'train':
		# Store the cache from the helper so backward pass can access it
		cache['stem'] = stem_cache 
		cache['flat_shape'] = (N, C, H, W)
		cache['flat'] = h_flat
		return probs, cache
		
	return probs

def value_head(x, params, mode='inference'):
	cache = {}
	
	# 1. Conv -> BN -> ReLU (using helper)
	h, stem_cache = conv_bn_relu_forward(
		x, 
		params['v_w1'], params['v_b1'],
		params['v_bn_g'], params['v_bn_b'], 
		params['v_bn_m'], params['v_bn_v'],
		pad=0,mode=mode
	)
	
	# 2. Flatten
	N, C, H, W = h.shape
	h_flat = h.reshape(N, -1)
	
	# 3. Dense Layer 1 + ReLU
	h1 = dense_forward(h_flat, params['v_fc1_w'], params['v_fc1_b'])
	h1_relu = relu(h1)
	
	# 4. Dense Layer 2 + Tanh
	h2 = dense_forward(h1_relu, params['v_fc2_w'], params['v_fc2_b'])
	value = tanh(h2)
	
	if mode == 'train':
		cache['stem'] = stem_cache
		cache['flat_shape'] = (N, C, H, W)
		cache['flat'] = h_flat
		cache['h1'] = h1 # Cached for ReLU backward
		cache['h1_relu'] = h1_relu # Cached for Dense backward
		return value, cache
		
	return value

# --- Backwards ---

def relu_backward(dout, x):
	"""d(Relu)/dx = 1 if x > 0 else 0"""
	dx = dout.copy()
	dx[x <= 0] = 0
	return dx

def dense_backward(dout, x, w, b):
	"""
	Backward for Fully Connected Layer.
	dout: (N, D_out)
	x: (N, D_in)
	w: (D_in, D_out)
	"""
	# dL/dW = x.T * dout
	dw = np.dot(x.T, dout)
	# dL/db = sum(dout)
	db = np.sum(dout, axis=0)
	# dL/dx = dout * w.T
	dx = np.dot(dout, w.T)
	
	return dx, dw, db

def tanh_backward(dout, out):
	"""d(tanh)/dx = 1 - tanh(x)^2"""
	return dout * (1 - out**2)


def conv2d_backward_slow(dout, x, w, b, stride=1, pad=1):
	"""
	Computes gradients for a convolutional layer.
	Fixed to handle pad=0 correctly.
	"""
	N, C, H, W = x.shape
	F, _, HH, WW = w.shape
	_, _, H_out, W_out = dout.shape
	
	# Initialize gradients
	dx = np.zeros_like(x)
	dw = np.zeros_like(w)
	db = np.zeros_like(b)
	
	# Pad input and dx
	# When pad=0, this just returns the array as-is
	x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
	dx_pad = np.pad(dx, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
	
	# Loop over all output points
	for n in range(N):
		for f in range(F):
			# db is simply the sum of gradients for that filter
			db[f] += np.sum(dout[n, f])
			
			for i in range(H_out):
				for j in range(W_out):
					vert_start = i * stride
					vert_end = vert_start + HH
					horiz_start = j * stride
					horiz_end = horiz_start + WW
					
					# 1. dw += x_slice * dout
					x_slice = x_pad[n, :, vert_start:vert_end, horiz_start:horiz_end]
					dw[f] += x_slice * dout[n, f, i, j]
					
					# 2. dx += w * dout
					dx_pad[n, :, vert_start:vert_end, horiz_start:horiz_end] += w[f] * dout[n, f, i, j]
	
	# Remove padding from dx to return to original shape (N, C, H, W)
	# FIX: Handle pad=0 case where pad:-pad would be 0:0 (empty)
	if pad == 0:
		dx = dx_pad
	else:
		dx = dx_pad[:, :, pad:-pad, pad:-pad]
	
	return dx, dw, db
	
def conv2d_backward(dout, x, w, b, stride=1, pad=1):
	"""
	Optimized Backward Convolutional Layer.
	Eliminates the Batch (N) and Filter (F) loops using np.tensordot.
	"""
	N, C, H, W = x.shape
	F, _, HH, WW = w.shape
	_, _, H_out, W_out = dout.shape
	
	# 1. db is trivial: just sum the gradient over the Batch, Height, and Width axes.
	# We keep the Filter (axis 1) separate.
	db = np.sum(dout, axis=(0, 2, 3))
	
	# Initialize padded dx and dw
	dx_pad = np.zeros((N, C, H + 2*pad, W + 2*pad))
	dw = np.zeros_like(w)
	
	x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
	
	# 2. Loop ONLY over the spatial output dimensions (9 iterations for 3x3)
	for i in range(H_out):
		for j in range(W_out):
			vert_start = i * stride
			vert_end = vert_start + HH
			horiz_start = j * stride
			horiz_end = horiz_start + WW
			
			# Extract slices
			x_slice = x_pad[:, :, vert_start:vert_end, horiz_start:horiz_end] # Shape: (N, C, HH, WW)
			dout_slice = dout[:, :, i, j] # Shape: (N, F)
			
			# --- dw Calculation ---
			# dw requires multiplying dout with x_slice, summing across the Batch (N) dimension.
			# We align axis 0 (N) of dout_slice with axis 0 (N) of x_slice.
			# Output naturally becomes (F, C, HH, WW), which matches dw perfectly!
			dw += np.tensordot(dout_slice, x_slice, axes=([0], [0]))
			
			# --- dx Calculation ---
			# dx requires multiplying dout with the weights w, summing across the Filters (F).
			# We align axis 1 (F) of dout_slice with axis 0 (F) of w.
			# Output naturally becomes (N, C, HH, WW), which drops perfectly into dx_pad!
			dx_pad[:, :, vert_start:vert_end, horiz_start:horiz_end] += np.tensordot(dout_slice, w, axes=([1], [0]))
	
	# 3. Strip padding from dx
	if pad == 0:
		dx = dx_pad
	else:
		dx = dx_pad[:, :, pad:-pad, pad:-pad]
	
	return dx, dw, db
	
def batch_norm_backward(dout, cache):
	"""
	Backward pass for batch normalization.
	Uses the cache tuple (x_norm, gamma, x_minus_mean, var, epsilon) saved during forward.
	Note: You need to modify your forward pass to return this cache!
	"""
	x_norm, gamma, x_minus_mean, var, epsilon = cache
	N, C, H, W = dout.shape
	
	# 1. Derivative w.r.t Beta and Gamma
	# Sum over N, H, W (all spatial dimensions + batch)
	dbeta = np.sum(dout, axis=(0, 2, 3))
	dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))
	
	# 2. Derivative w.r.t Input X
	# This derivation is lengthy, but the optimized formula is:
	# dx = (1 / M) * gamma * (var)^-1/2 * (M * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
	# where M = N * H * W
	
	M = N * H * W
	
	# Reshape for broadcasting
	gamma_reshaped = gamma.reshape(1, C, 1, 1)
	var_reshaped = var.reshape(1, C, 1, 1)
	
	# Gradient of the normalized input
	dx_norm = dout * gamma_reshaped
	
	# Standard deviation inverse
	std_inv = 1.0 / np.sqrt(var_reshaped + epsilon)
	
	# The Complex Part (dL/dMean and dL/dVar combined)
	dx = (1.0 / M) * std_inv * (
		M * dx_norm - 
		np.sum(dx_norm, axis=(0, 2, 3), keepdims=True) - 
		x_norm * np.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True)
	)
	
	return dx, dgamma, dbeta


def conv_bn_relu_backward(dout, cache, w, b, pad):
	grads = {}
	d_relu = relu_backward(dout, cache['relu_in'])
	d_bn, grads['bn_g'], grads['bn_b'] = batch_norm_backward(d_relu, cache['bn_cache'])
	dx, grads['w'], grads['b'] = conv2d_backward(d_bn, cache['x'], w, b, pad=pad)
	return dx, grads

def residual_block_backward(dout, cache, params):
	"""
	Computes gradients for a Residual Block.
	"""
	grads = {}
	
	# Unpack Cache
	x = cache['x']
	bn1_cache = cache['bn1_cache']
	bn1_out = cache['bn1_out']
	r1_out = cache['r1_out']
	bn2_cache = cache['bn2_cache']
	add_out = cache['add_out']
	
	# 7. Final ReLU Backward
	d_add = relu_backward(dout, add_out)
	
	# 6. Skip Connection Backward
	# The gradient splits: one part to the block (d_bn2_out), one to shortcut (d_shortcut)
	d_bn2_out = d_add 
	d_shortcut = d_add 
	
	# 5. BN2 Backward
	d_c2_out, grads['bn2_g'], grads['bn2_b'] = batch_norm_backward(d_bn2_out, bn2_cache)
	
	# 4. Conv2 Backward
	d_r1_out, grads['w2'], grads['b2'] = conv2d_backward(d_c2_out, r1_out, params['w2'], params['b2'])
	
	# 3. ReLU1 Backward
	d_bn1_out = relu_backward(d_r1_out, bn1_out)
	
	# 2. BN1 Backward
	d_c1_out, grads['bn1_g'], grads['bn1_b'] = batch_norm_backward(d_bn1_out, bn1_cache)
	
	# 1. Conv1 Backward
	dx_block, grads['w1'], grads['b1'] = conv2d_backward(d_c1_out, x, params['w1'], params['b1'])
	
	# Combine gradients (Input x sent to Conv1 + Input x sent to Skip)
	dx = dx_block + d_shortcut
	
	return dx, grads

def policy_head_backward(dout, cache, params):
	"""
	Backwards pass for Policy Head.
	dout: Gradient of Loss w.r.t logits (N, num_moves)
	"""
	grads = {}
	
	# 1. Backprop Dense Layer
	# dL/dw = x.T * dout
	dx_flat, grads['p_fc_w'], grads['p_fc_b'] = dense_backward(
		dout, cache['flat'], 
		params['p_fc_w'], params['p_fc_b']
	)
	
	# 2. Unflatten (N, C*H*W) -> (N, C, H, W)
	d_stem = dx_flat.reshape(cache['flat_shape'])
	
	# 3. Backprop Stem (Conv -> BN -> ReLU)
	# Note: pad=0 as defined in forward pass
	dx, stem_grads = conv_bn_relu_backward(
		d_stem, cache['stem'],
		params['p_w1'], params['p_b1'], pad=0
	)
	
	# Map the generic stem gradients ('w', 'b') to specific param keys ('p_w1', etc.)
	grads['p_w1'] = stem_grads['w']
	grads['p_b1'] = stem_grads['b']
	grads['p_bn_g'] = stem_grads['bn_g']
	grads['p_bn_b'] = stem_grads['bn_b']
	
	return dx, grads

def value_head_backward(d_v_logits, cache, params):
	"""
	Backwards pass for Value Head.
	d_v_logits: Gradient entering the head (dL/d_tanh_output * d_tanh/d_h2)
	"""
	grads = {}
	
	# 1. Backprop Dense 2
	dh1_relu, grads['v_fc2_w'], grads['v_fc2_b'] = dense_backward(
		d_v_logits, cache['h1_relu'],
		params['v_fc2_w'], params['v_fc2_b']
	)
	
	# 2. Backprop ReLU
	dh1 = relu_backward(dh1_relu, cache['h1'])
	
	# 3. Backprop Dense 1
	dx_flat, grads['v_fc1_w'], grads['v_fc1_b'] = dense_backward(
		dh1, cache['flat'],
		params['v_fc1_w'], params['v_fc1_b']
	)
	
	# 4. Unflatten
	d_stem = dx_flat.reshape(cache['flat_shape'])
	
	# 5. Backprop Stem
	dx, stem_grads = conv_bn_relu_backward(
		d_stem, cache['stem'],
		params['v_w1'], params['v_b1'], pad=0
	)
	
	# Map generic keys to specific param keys
	grads['v_w1'] = stem_grads['w']
	grads['v_b1'] = stem_grads['b']
	grads['v_bn_g'] = stem_grads['bn_g']
	grads['v_bn_b'] = stem_grads['bn_b']
	
	return dx, grads
	
# --- 1. Parameter Initialization (The "Brain") ---
def init_random_params(board_size=3, filters=16):
	input_planes = 17
	num_moves = board_size * board_size + 1 
	
	# Xavier/He Initialization helper
	def init_w(shape): return np.random.randn(*shape) * np.sqrt(2. / np.prod(shape[1:]))
	
	params = {
		# --- 1. The Stem (Initial Conv) ---
		'stem_w': init_w((filters, input_planes, 3, 3)), 'stem_b': np.zeros(filters),
		'stem_bn_g': np.ones(filters), 'stem_bn_b': np.zeros(filters),
		'stem_bn_m': np.zeros(filters), 'stem_bn_v': np.ones(filters),

		# --- 2. Residual Block ---
		'res_w1': init_w((filters, filters, 3, 3)), 'res_b1': np.zeros(filters),
		'res_bn1_g': np.ones(filters), 'res_bn1_b': np.zeros(filters),
		'res_bn1_m': np.zeros(filters), 'res_bn1_v': np.ones(filters),
		
		'res_w2': init_w((filters, filters, 3, 3)), 'res_b2': np.zeros(filters),
		'res_bn2_g': np.ones(filters), 'res_bn2_b': np.zeros(filters),
		'res_bn2_m': np.zeros(filters), 'res_bn2_v': np.ones(filters),
		
		# --- 3. Heads ---
		# Policy Head
		'p_w1': init_w((2, filters, 1, 1)), 'p_b1': np.zeros(2),
		'p_bn_g': np.ones(2), 'p_bn_b': np.zeros(2),
		'p_bn_m': np.zeros(2), 'p_bn_v': np.ones(2),
		'p_fc_w': init_w((2*board_size*board_size, num_moves)), 'p_fc_b': np.zeros(num_moves),
		
		# Value Head
		'v_w1': init_w((1, filters, 1, 1)), 'v_b1': np.zeros(1),
		'v_bn_g': np.ones(1), 'v_bn_b': np.zeros(1),
		'v_bn_m': np.zeros(1), 'v_bn_v': np.ones(1),
		'v_fc1_w': init_w((1*board_size*board_size, 16)), 'v_fc1_b': np.zeros(16),
		'v_fc2_w': init_w((16, 1)), 'v_fc2_b': np.zeros(1)
	}
	return params

def run_neural_net(game, params):
	# 1. Encode Input
	input_tensor = encode_input_tensor(game) 
	x = input_tensor[np.newaxis, :, :, :] # (1, 17, 3, 3)
	
	# 2. The Stem
	out = conv2d_forward(x, params['stem_w'], params['stem_b'], stride=1, pad=1)
	# Note: inference doesn't need cache, so we ignore the second return value
	out, _ = batch_norm_forward(out, params['stem_bn_g'], params['stem_bn_b'], 
							 params['stem_bn_m'], params['stem_bn_v'],mode='inference')
	out = relu(out)
	
	# 3. Residual Tower
	res_params = {
		'w1': params['res_w1'], 'b1': params['res_b1'],
		'bn1_g': params['res_bn1_g'], 'bn1_b': params['res_bn1_b'],
		'bn1_m': params['res_bn1_m'], 'bn1_v': params['res_bn1_v'],
		'w2': params['res_w2'], 'b2': params['res_b2'],
		'bn2_g': params['res_bn2_g'], 'bn2_b': params['res_bn2_b'],
		'bn2_m': params['res_bn2_m'], 'bn2_v': params['res_bn2_v'],
	}
	
	# FIX: Unpack the tuple! Ignore the cache (_)
	out, _ = residual_block_forward(out, res_params)
	
	# 4. Heads
	policy_probs = policy_head(out, params)
	value_est = value_head(out, params)
	
	return policy_probs[0], value_est[0][0]
