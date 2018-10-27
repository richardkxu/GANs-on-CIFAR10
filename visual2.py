import os

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import Discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 100

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

# discriminator trained without the generator
model =  Discriminator()
checkpoint = torch.load('./checkpoint/discriminator-run-20181025021504/discriminator.model')
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()

# discriminator trained with the generator
oldmodel = torch.load('./checkpoint/gan-run-20181025172252/discriminator.model')
state_dict = oldmodel.state_dict()
model2 = Discriminator()
model2.load_state_dict(state_dict)
model2.cuda()
model2.eval()

############## Perturb Real Images ##############

# Grab a sample batch from the test dataset
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()

# max feat plot for discriminator without the generator
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/discri_max_features_layer2.png', bbox_inches='tight')
plt.close(fig)


# max feat plot for discriminator with the generator
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model2(X)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/gan_max_features_layer2.png', bbox_inches='tight')
plt.close(fig)


