import torch
import torchvision
import torchvision.transforms as transforms
from resnet_models.normal_resnet import NormalResNet
from resnet_models.sharing_resnet import SharingResNet
from resnet_models.memory_resnet import MemoryResNet
import time
import os

MAX_STEP = 50000
BATCH_SIZE = 256

model_dict = {"normal_resnet": NormalResNet, "sharing_resnet": SharingResNet, "memory_resnet": MemoryResNet}
# model_name = "normal_resnet"
# model_name = "sharing_resnet"
model_name = "memory_resnet"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"device = {device}")

data_root = "./data"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

trainset = torchvision.datasets.MNIST(root=data_root,
                                      train=True,
                                      download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.MNIST(root=data_root,
                                     train=False,
                                     download=True,
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=2)

model = model_dict[model_name](1, 128, 10, 4)
model.to(device)
crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

result_dir = "./result_mnist"
os.makedirs(result_dir, exist_ok=True)
result_file = open(f"{result_dir}/train_log_{model_name}.tsv", "w")

start = time.time()

EPOCH = 10
for e in range(EPOCH):
    for i, (x, label) in enumerate(trainloader, 1):
        x = x.to(device)
        label = label.to(device)
        logit = model.forward(x)

        loss = crit(logit, label)

        # 正答率の計算
        pred = torch.argmax(logit, dim=1)
        compare = (pred == label)
        accuracy = (compare.sum().float() / compare.numel())

        elapsed = time.time() - start
        loss_str = f"{elapsed:5.1f}\t{i:4d}\t{loss.item():.4f}\t{accuracy.item():.4f}"
        print(loss_str, end="\r")

        if i % (MAX_STEP // 25) == 0:
            print()
            result_file.write(loss_str + "\n")

        optim.zero_grad()
        loss.backward()
        optim.step()
    print()
