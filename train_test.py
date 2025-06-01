import torch, torch.nn as nn, torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import time, os


def main():
    print("torch.version.cuda  :", torch.version.cuda)  # 编译期用的 CUDA 主版本号
    print("torch.version.git   :", torch.version.git_version)  # 方便确认 wheel 来源
    print("cuda.is_available() :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Driver capability :", torch.cuda.get_device_capability(0))
        print("Device name       :", torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    bs = 128
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    i=0
    while(i<100):
        print(device)
        i+=1
        time.sleep(1)

    # data_path = "./datasets/data"
    #
    # train_set = dsets.CIFAR10(root=data_path, train=True, transform=tf, download=False)
    # test_set = dsets.CIFAR10(root=data_path, train=False, transform=tf, download=False)
    #
    # train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4)
    #
    # model = resnet18(num_classes=10).to(device)
    # opt = optim.Adam(model.parameters(), lr=1e-3)
    # lossf = nn.CrossEntropyLoss()
    #
    # for epoch in range(100):
    #     model.train()
    #     t0 = time.time()
    #     running = 0
    #     for x, y in train_loader:
    #         x, y = x.to(device), y.to(device)
    #         opt.zero_grad()
    #         loss = lossf(model(x), y)
    #         loss.backward()
    #         opt.step()
    #         running += loss.item()
    #     print(f"Epoch {epoch:02d}  train-loss {running / len(train_loader):.3f} "
    #           f"time {time.time() - t0:.1f}s")
    #
    # # 简单验证
    # model.eval()
    # correct = 0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         x, y = x.to(device), y.to(device)
    #         correct += (model(x).argmax(1) == y).sum().item()
    # print(f"Test accuracy: {correct / len(test_set):.3%}, total time {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
