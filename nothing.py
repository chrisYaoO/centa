from train_dist import *

import matplotlib.pyplot as plt

plt.plot(timeline, train_acc_list, label='Train Accuracy')
plt.plot(timeline, test_acc_list, label='Test Accuracy')
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



