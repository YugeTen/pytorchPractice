import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'images'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))])
mnist = torchvision.datasets.MNIST(root = 'data/',
                                  train = True,
                                  transform = transform,
                                  download = True)
data_loader = torch.utils.data.DataLoader(dataset = mnist,
                                         batch_size = batch_size,
                                         shuffle = True)

# discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2), #0.2: Controls the angle of the negative slope
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# device setting
D = D.to(device)
G = G.to(device)

# loss and optimizer
criterion = nn.BCELoss() # binary cross entropy loss
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1) #min=0, max=1

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        # create the labels that are later used as input for BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)  # the 1s
        fake_labels = torch.zeros(batch_size, 1).to(device)  # the 0s

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        # loss = - y * log(D(x)) - (1-y) * log(1 - D(x))

        # real loss:
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs # just for display later

        # fake loss:
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs # for display later

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # Save real images
    if (epoch + 1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

# Save the model checkpoints
torch.save(G, 'model/8GAN_yunjey_G.pt')
torch.save(D, 'model/8GAN_yunjey_D.pt')





















