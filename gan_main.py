import os
import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import Discriminator, Generator

# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='which gpu(cuda visible device) to use')
args = parser.parse_args()

if not args.gpu:
    print("Using all available GPUs, data parallelism")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    print("Using gpu: {}".format(args.gpu))

batch_size = 128
n_epoch = 250
learning_rate = 0.0001
n_classes = 10
gen_train = 1
n_z= 100  # dim of the random noise vector

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

# define model
aD =  Discriminator()
aD.cuda()

aG = Generator()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=learning_rate, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=learning_rate, betas=(0, 0.9))
criterion = nn.CrossEntropyLoss()


def annotate_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return name


if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
ckpt_dir = 'gan'
ckpt_dir = annotate_dir(ckpt_dir)
ckpt_dir = os.path.join('./checkpoint', ckpt_dir)
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

# creat dir for generated images
out_dir = os.path.join(ckpt_dir, 'output')
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# This function is used to plot a 10 by 10 grid of images scaled between 0 and 1
def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


def train(trainloader):
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if (Y_train_batch.shape[0] < batch_size):
            continue

        ################## train the Generator ##################
        # the generator is trained every gen_train number of iteration
        # Sometimes the discriminator is trained more frequently than the
        # generator meaning gen_train can be set to something like 5.
        if ((batch_idx % gen_train) == 0):
            # The gradients for the discriminator parameters are turned off
            # during the generator update as this saves GPU memory.
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0, n_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, n_z))
            label_onehot = np.zeros((batch_size, n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            # fake images coming from the generator
            fake_data = aG(noise)
            # gen_source is the value from fc1, gen_class output is from fc10
            gen_source, gen_class = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()

        ################## train the Discriminator ##################
        # turn on gradient for discriminator, since now we need to update it
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # train discriminator with input from generator
        label = np.random.randint(0, n_classes, batch_size)
        noise = np.random.normal(0, 1, (batch_size, n_z))
        label_onehot = np.zeros((batch_size, n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()  # fc1 loss
        disc_fake_class = criterion(disc_fake_class, fake_label)  # fc10 loss

        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = (float(prediction.eq(real_label.data).sum()) / float(batch_size)) * 100.0

        disc_real_source = disc_real_source.mean()  # fc1 loss
        disc_real_class = criterion(disc_real_class, real_label)  # fc10 loss

        gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()

        optimizer_d.step()


def test(testloader):
    # Test the model
    aD.eval()

    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch = Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1]  # first column has actual prob.
            accuracy = (float(prediction.eq(Y_test_batch.data).sum()) / float(batch_size)) * 100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)

    return accuracy_test


################## driver code ##################

# creat the noise vectors used to generate images
# same noise vectors are used for every epoch for comparision
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100, n_z))  # 100x100 random noise 0-1
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1  # 100x10 one hot label, each row one label
# each row is a 100 dim vector that will be transformed into fake image
# set first 10 number of each row as one hot label
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

for epoch in range(n_epoch):

    # turn model to train mode
    aG.train()
    aD.train()

    time1 = time.time()
    train(trainloader)
    accuracy_test = test(testloader)
    time2 = time.time()

    sec = time2-time1
    min, sec = divmod(sec, 60)
    hr, min = divmod(min, 60)
    print('Epoch: {} | Test Acc: {:.3f}% | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(epoch, accuracy_test, hr, min, sec))

    # generate images using save_noise created before training
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0, 2, 3, 1)
        aG.train()

    fig = plot(samples)
    plt.savefig(out_dir + '/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if (((epoch + 1) % 50) == 0):
        torch.save(aG, os.path.join(out_dir, 'tempG_' + str(epoch) + '.model'))
        torch.save(aD, os.path.join(out_dir, 'tempD_' + str(epoch) + '.model'))

# save final model
torch.save(aG, os.path.join(out_dir, 'generator.model'))
torch.save(aD, os.path.join(out_dir, 'discriminator.model'))

