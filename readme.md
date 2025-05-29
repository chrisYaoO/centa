Install cvx packages on Matlab at http://cvxr.com

First run 'generate_graphs.m' to obtain the network information, including the topology, comm. and comp. time
    (In this folder, I have included two generated traces
        graphs_iter1_8workers_range60_0805.mat
        graphs_iter1_50workers_range60_0805.mat)

Second, run 'main_CENT_0805.m' to obtain a sparse network topology, including the consensus weight matrix and bandwidth allocation matrix
    (In this folder, I have included two generated traces
        CENT_solutions_iter1_8workers_range60.mat
        CENT_solutions_iter1_50workers_range60.mat)

Third, incorporate the consensus weight matrix to PyTorch distributed learning
    Install 'more_itertools' with 'conda install -c anaconda more-itertools'

    1) run simulation: simulation_train_dist.py
        2) run emulation: train_dist.py (use torch.distributed and torch.multiprocessing)

Note: the communication and computation time are emulated in 'generate_graphs.m'. 
'simulation_train_dist.py' and 'train_dist.py' are only used to see the training performance in each training round.







Toolbox failed start  and without bandwidth constraint

1. create github

2. use matlab  to get consensus weight matrix (can't do it now)

3. put the code matrix

4. use torchrun to start training

   1. split the data

   1. nccl communication without builiding  network topology ok? just update it using weight matrix

   1. for each epoch communicate to share parameters and update 

2. get result at .out


### Todo:
1. put datasets to share location or sync to local_scratch before training otherwise it may affect the speed of io since multi node multi process
2. 4 nodes first 

