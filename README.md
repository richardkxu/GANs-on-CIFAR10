# Generative Adversarial Network

* Utilize techniques from ACGAN and Wasserstein GANs

* BN and ReLU are applied after transposed conv as well as conv in generator

* For transposed conv, use torch.nn.ConvTranspose2d. Transposed conv first blow up input region by inserting 0s 
between adjacent pixels. Then perform a conv operation

* 0.5 mean and 0.5 std normalization is applied on training images. This change pixel range from 0 to 1 to -1 to 1. 
However, the output of generator is 0 to 1. So use tanh fn to scale to -1 to 1 in order to match the input 
range of the discriminator

* discriminator contains fc1 and fc10 as the output:
```python
out1 = self.fc1(x)
out10 = self.fc10(x)
return out1, out10
```

* use 196 intermediate featmaps and train for 200 epochs (tutorial train for 500 epochs)

* random noise 1d vector of 100,1 is used as starting point for generator. 1st 10 numbers are the one-hot 
encoding for the target class of this fake image. The generator transform the 100 noise vector into a fake image

* Discriminator without Generator: regular resnet to train

* Discriminator with Generator: GAN algorithm

