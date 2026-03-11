# AlphaZero from Scratch (Phase 1/3): Pure NumPy 3x3 Go

This repository contains **Phase 1** of a three-part project designed to build a complete, production-grade AlphaZero game engine. 

This phase purposefully avoids black-box APIs (like `torch.autograd` or standard `nn.Conv2d`) to implement an AlphaZero like algorithm—including Monte Carlo Tree Search (MCTS), Batch Normalization, and Convolutional backpropagation—entirely from scratch using pure Python and NumPy.

### Project Portfolio Context
* **Phase 1 (This Repo): Mathematical Fundamentals.** Pure NumPy 3x3 Go. Demonstrates manual backpropagation, tree search, and algorithm architecture.
* **Phase 2: Systems & Optimization.** Custom C++/CUDA kernels for Chess. Demonstrates zero-copy memory management and hardware-level optimization.
* **Phase 3: Production Scaling.** PyTorch Chess Engine. Demonstrates industry-standard scaling, high-throughput data pipelines, and distributed training.

## Key Technical Features

* **Hand-Rolled Backpropagation:** Implemented manual forward and backward passes for Dense, Conv2D, and Batch Normalization layers. 
* **Tensor Optimization:** Translated naive O(N * F * H * W) nested-loop convolutions into optimized `np.tensordot` operations, bridging the gap between mathematical formulation and C-level execution speed.
* **Monte Carlo Tree Search (MCTS):** Implemented PUCT (Predictor + Upper Confidence Bound applied to Trees) for tree traversal, balancing exploration (via Dirichlet noise) and exploitation (via Neural Network priors).
* **D4 Dihedral Group Augmentation:** Implemented mathematical symmetries (4 rotations, 4 reflections) to augment the replay buffer, maximizing sample efficiency in a deterministic environment.
* **Momentum SGD:** Custom optimizer with gradient clipping and L2 weight decay (regularization).

## Architecture

To maintain a strict separation of concerns, the codebase is modularized:

* `environment.py`: Contains the `RealGoGame` class. Manages the board state, legal moves, Ko rule checks, and territory scoring (Area Scoring).
* `nn_numpy.py`: The "Brain". Contains the ResNet architecture, state-to-tensor encoding, and the manual gradient calculus.
* `mcts.py`: The "Search". Handles the tree-based lookahead and node expansion.
* `trainer.py`: The "Loop". Orchestrates self-play data generation, symmetry augmentation, and mini-batch SGD updates.

## Results & Convergence

3x3 Go was chosen for this pure-NumPy phase because it has a known optimal strategy (playing in the center, `(1, 1)`), allowing for clear verification of network convergence without requiring GPU days of training.

After 500 games of self-play (a buffer size of ~10,000 states), the network correctly evaluates the empty board as a forced win for Black and outputs a policy probability distribution overwhelmingly favoring the mathematical optimum: the center node.

```text
Network Evaluation of Empty Board: 0.9384
>> VERDICT: Correctly evaluates Black advantage.

Policy Probability Map:
[[0.001 0.002 0.   ]
 [0.002 0.992 0.002]
 [0.    0.002 0.   ]]
>> SUCCESS: The AI prefers the Center (1,1).
```

## How to run

Clone the repository.

Install requirements: pip install numpy matplotlib seaborn

Run the Jupyter Notebook 3x3_Go.ipynb to execute the self-play loop, train the network, and visualize the policy map.
