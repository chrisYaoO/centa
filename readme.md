# CENTA+Beluga Cluster

### Matlab instructions by Jingrong
1. Install cvx packages on Matlab at http://cvxr.com

2. run 'generate_graphs.m' to obtain the network information, including the topology, comm. and comp. time

    (In this folder, I have included two generated traces
        graphs_iter1_8workers_range60_0805.mat
        graphs_iter1_50workers_range60_0805.mat)
3. run 'main_CENT_0805.m' to obtain a sparse network topology, including the consensus weight matrix and bandwidth allocation matrix
    
    (In this folder, I have included two generated traces
        CENT_solutions_iter1_8workers_range60.mat
        CENT_solutions_iter1_50workers_range60.mat)

4. incorporate the consensus weight matrix to PyTorch distributed learning 
   1. Install 'more_itertools' with 'conda install -c anaconda more-itertools' 
   2. run simulation: simulation_train_dist.py 
   3. run emulation: train_dist.py (use torch.distributed and torch.multiprocessing)

5. Note: the communication and computation time are emulated in 'generate_graphs.m'. 
'simulation_train_dist.py' and 'train_dist.py' are only used to see the training performance in each training round.



### Current
1. I understand that W is obtained via FDLA and the link bandwidths via the min-max allocation. In the actual simulation, did you enforce these bandwidth constraints by
introducing artificial time delays during gradient exchange, or
directly modifying the communication bandwidth (e.g., throttling link rates)?
2. On Compute Canada clusters, indeed I cannot control InfiniBand link rates, so I currently inject delays after each gradient-averaging step based on the slowest allocated bandwidth. Would you recommend continuing with my delay-injection approach, or simply omitting the bandwidth constraints entirely?
3. In the actual simulation, was the pre-calculated consensus weight matrix W applied to average gradients once per batch or once per epoch?
4. Is the niid partitioning in the simulation code identical to what you used for the paper (i.e., controlling the number of classes per device via p)?
5. In the provided simulation code, gradient averaging across models is conducted once per epoch when using heterogeneous datasets. However, my understanding is that it would typically be done per batch. I assume the per-epoch is intentional, possibly because the niid partition function results in unequal batch counts across devices, which could cause synchronization issues. To address this, I modified the function to ensure the same number of batches per device while preserving the niid nature of the data—though it comes at the cost of using fewer training samples. Is it acceptable to use my modified niid partition function for the reproduction?

### Todo:
1. reproduce using same settings

### instructions