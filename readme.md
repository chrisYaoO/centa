# CENTA+beluga cluster

### Matlab instructions by jinrong
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
1. waiting for test result
2. centa implementaiton

### Todo:
1. run tests
2. plot

### instructions




