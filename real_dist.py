from train_dist import *
import os, argparse, torch, torch.distributed as dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backend", type=str, default="nccl")  # GPU: nccl, CPU: gloo
    parser.add_argument("--w_type", type=int, default=5)  # 1~7
    args = parser.parse_args()


    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


    W_type = args.w_type
    n_epoch = args.epochs
    range_ = 60
    case = 1
    size = 8

    filename = 'fed/'+'CENT_solutions_iter1_' + str(size) + 'workers_range' + str(range_) + '.mat'

    if W_type == 1:  # md
        W = sio.loadmat(filename)['W_md']
    elif W_type == 2:  # mh
        W = sio.loadmat(filename)['W_mh']
    elif W_type == 3:  # bc
        W = sio.loadmat(filename)['W_bc']
    elif W_type == 4:  # fdla
        W = sio.loadmat(filename)['W_fdla']
    elif W_type == 5:  # cent
        W = sio.loadmat(filename)['W_cent']
    elif W_type == 6:  # fully connected
        W = np.ones((int(size), int(size))) * (1 / size)
    else:  # no communication
        W = np.identity(size)

    W=W.cuda()
    # print(W)

    torch.manual_seed(10 * dist.get_rank())  # 4321)#1234)

    # here i iid data, can also use the non-iid case as in the simulation code
    file_path='fed/datasets'
    train_set, bsz = partition_dataset(file_path)
    test_set, test_bsz = partition_dataset_test()

    """ Train """
    model = create_lenet().cuda()  # Net() #LeNet()#MLP()#AlexNet()#
    # model = model
    # model = model.cuda(rank)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cec = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    # num_batches = ceil(len(train_set.dataset) / float(bsz))
    batch_loss = []
    batch_accuracy = []
    run_time = []
    run_comp_time = []

    batch_test_loss = []
    batch_test_accuracy = []
    run_test_time = []

    iter_count = 0
    for epoch in range(n_epoch):

        if dist.get_rank() == 3:
            print('epoch =', epoch)

        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_idx = 0

        for data, target in train_set:
            batch_idx += 1

            data, target = Variable(data), Variable(target)
            # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = cec(output, target)

            epoch_loss += loss.item()  # data[0]
            loss.backward()

            average_gradients(model, W)

            optimizer.step()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            """append data"""
            batch_loss.append(epoch_loss / (batch_idx + 1))
            batch_accuracy.append(100. * correct / total)

            if dist.get_rank() == 3:
                if batch_idx % int(len(train_set) / 5) == 0:
                    print(batch_idx, len(train_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (epoch_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            iter_count += 1

        if dist.get_rank() == 3:
            """ Test """
            test_epoch_loss = 0.0
            test_correct = 0
            test_total = 0
            test_batch_idx = 0

            # test_start_time = time.time()

            for data, target in test_set:
                test_batch_idx += 1

                data, target = Variable(data), Variable(target)

                output = model(data)
                loss = F.nll_loss(output, target)
                test_epoch_loss += loss.item()  # data[0]
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                print(test_batch_idx, len(test_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_epoch_loss / (test_batch_idx + 1), 100. * test_correct / test_total, test_correct,
                         test_total))

            """append data"""
            batch_test_loss.append(test_epoch_loss / (test_batch_idx + 1))
            batch_test_accuracy.append(100. * test_correct / test_total)

    if dist.get_rank() == 3:
        mdic = {'batch_training_loss': batch_loss,
                'batch_training_accuracy': batch_accuracy,
                'run_training_time': run_time,
                'run_comp_time': run_comp_time,
                'batch_test_loss': batch_test_loss,
                'batch_test_accuracy': batch_test_accuracy,
                'run_test_time': run_test_time,
                'n_epoch': n_epoch,
                'worker_ID': dist.get_rank()}
        for k, v in mdic.items():
            print(k, v)
        # if W_type == 1:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Max-degree_range" + str(range_) + str(case) + ".mat", mdic)
        # elif W_type == 2:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Metropolis_range" + str(range_) + str(case) + ".mat", mdic)
        # elif W_type == 3:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Best-constant_range" + str(range_) + str(case) + ".mat", mdic)
        # elif W_type == 4:
        #     sio.savemat(
        #         "worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(n_epoch) + "_epochs_FDLA_range" + str(
        #             range_) + str(case) + ".mat", mdic)
        # elif W_type == 5:
        #     sio.savemat(
        #         "worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(n_epoch) + "_epochs_CENT_range" + str(
        #             range_) + str(case) + "_static.mat", mdic)
        # elif W_type == 6:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Fully-connected_range" + str(range_) + str(case) + ".mat", mdic)
        # else:
        #     sio.savemat(
        #         "worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(n_epoch) + "_epochs_No-consensus.mat",
        #         mdic)
