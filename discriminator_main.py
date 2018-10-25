import os
import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from model import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='which gpu(cuda visible device) to use')
args = parser.parse_args()

if not args.gpu:
    print("Using all available GPUs, data parallelism")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    print("Using gpu: {}".format(args.gpu))
device = 'cuda'

batch_size = 128
n_epoch = 100
learning_rate = 0.0001

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model =  Discriminator()
model.to(device)
if not args.gpu:
    net = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def annotate_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return name

# driver code
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
ckpt_dir = 'discriminator'
ckpt_dir = annotate_dir(ckpt_dir)
ckpt_dir = os.path.join('./checkpoint', ckpt_dir)
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

best_accuracy = 0.0


def train(epoch, trainloader):

    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    time1 = time.time()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        X_train_batch = X_train_batch.to(device)
        Y_train_batch = Y_train_batch.to(device)
        # only need fc10 output
        _, fc10_out = model(X_train_batch)

        loss = criterion(fc10_out, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * Y_train_batch.size(0)
        _, predicted = fc10_out.max(1)
        total += Y_train_batch.size(0)
        correct += predicted.eq(Y_train_batch).sum().item()

    total_loss = train_loss / len(trainloader.dataset)
    total_acc = 100.0 * correct / total
    time2 = time.time()
    sec = time2-time1
    min, sec = divmod(sec, 60)
    hr, min = divmod(min, 60)
    print('Epoch: {} | Train Loss: {:.3f} | Train Acc: {:.3f}% | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(epoch, total_loss, total_acc, hr, min, sec))


def test(epoch, testloader):
    global best_accuracy

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    time1 = time.time()
    with torch.no_grad():
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch = X_test_batch.to(device)
            Y_test_batch = Y_test_batch.to(device)
            # only need fc10 output
            _, fc10_out = model(X_test_batch)

            loss = criterion(fc10_out, Y_test_batch)

            test_loss += loss.item() * Y_test_batch.size(0)
            _, predicted = fc10_out.max(1)
            total += Y_test_batch.size(0)
            correct += predicted.eq(Y_test_batch).sum().item()

    total_loss = test_loss / len(testloader.dataset)
    total_acc = 100.0 * correct / total
    time2 = time.time()
    sec = time2-time1
    min, sec = divmod(sec, 60)
    hr, min = divmod(min, 60)
    print('Epoch: {} | Test Loss: {:.3f} | Test Acc: {:.3f}% | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(epoch, total_loss, total_acc, hr, min, sec))

    if total_acc > best_accuracy:
        best_accuracy = total_acc
        print("Saving ckpt at {}-th epoch.".format(epoch))
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'best_accuracy': best_accuracy,
            'best_loss': total_loss,
            'opt_state': optimizer.state_dict()
        }
        torch.save(state, os.path.join(ckpt_dir, 'discriminator.model'))


# driver code
for epoch in range(n_epoch):
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 10.0
    if epoch == 75:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 100.0

    train(epoch, trainloader)
    test(epoch, testloader)

