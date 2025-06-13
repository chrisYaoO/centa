from train_dist import *
import numpy as np
import matplotlib.pyplot as plt


# plt.plot(timeline, train_acc_list, label='Train Accuracy')
# plt.plot(timeline, test_acc_list, label='Test Accuracy')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy vs. Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

def load_matrix(filename, size: int, W_type=5):
    if W_type == 1:  # md
        W = sio.loadmat(filename)['W_md']
        B = sio.loadmat(filename)['B_md']
    elif W_type == 2:  # mh
        W = sio.loadmat(filename)['W_mh']
        B = sio.loadmat(filename)['B_mh']
    elif W_type == 3:  # bc
        W = sio.loadmat(filename)['W_bc']
        B = sio.loadmat(filename)['B_bc']
    elif W_type == 4:  # fdla
        W = sio.loadmat(filename)['W_fdla']
        B = sio.loadmat(filename)['B_fdla']
    elif W_type == 5:  # cent
        W = sio.loadmat(filename)['W_cent']
        B = sio.loadmat(filename)['B_cent']
    elif W_type == 6:  # fully connected
        W = np.ones((int(size), int(size))) * (1 / size)
        B = np.ones((int(size), int(size))) * 20 / (size * size)
    else:  # no communication
        W = np.identity(size)
        B = np.zeros(int(size) * int(size))

    B_ij = np.zeros((size, size))
    if np.count_nonzero(W) != len(B) + size:
        raise ValueError('size of W and B do not match')
    else:
        k = 0
        for i in range(size):
            for j in range(size):
                if W[i][j] != 0 and i != j:
                    B_ij[i][j] = B[k]
                    k += 1

    return W, B_ij


filename = 'data/CENT_solutions_iter1_4workers_range60.mat'
W, B_ij = load_matrix(filename, 4, 5)
print(B_ij)
