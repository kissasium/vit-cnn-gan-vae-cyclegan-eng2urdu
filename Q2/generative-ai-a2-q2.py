import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
from torchvision.utils import save_image

class SketchPhotoDataset(Dataset):
    def __init__(self, sketches_dir, photos_dir, transform=None):
        self.sketches_dir = sketches_dir
        self.photos_dir = photos_dir
        self.transform = transform
        self.sketches = os.listdir(sketches_dir)
    
    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketches_dir, self.sketches[idx])
        photo_path = os.path.join(self.photos_dir, self.sketches[idx])
        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)
        
        return sketch, photo

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class CycleGAN(nn.Module):
    def __init__(self, generator_ab, generator_ba, discriminator_a, discriminator_b):
        super(CycleGAN, self).__init__()
        self.generator_ab = generator_ab
        self.generator_ba = generator_ba
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b

    def forward(self, x):
        pass

def train_cyclegan(start_epoch=0, num_epochs=200):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = SketchPhotoDataset(
        sketches_dir='archive/train/sketches',
        photos_dir='archive/train/photos',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    generator_ab = Generator(input_channels=3, output_channels=3)
    generator_ba = Generator(input_channels=3, output_channels=3)
    discriminator_a = Discriminator(input_channels=6)
    discriminator_b = Discriminator(input_channels=6)

    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(list(generator_ab.parameters()) + list(generator_ba.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d_a = optim.Adam(discriminator_a.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d_b = optim.Adam(discriminator_b.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(start_epoch, num_epochs + start_epoch):
        for i, (sketches, photos) in enumerate(train_loader):
            sketches, photos = sketches, photos

            fake_photos = generator_ab(sketches)
            fake_sketches = generator_ba(photos)
            real_inputs_a = torch.cat((sketches, photos), 1)
            fake_inputs_a = torch.cat((sketches, fake_photos), 1)
            real_inputs_b = torch.cat((photos, sketches), 1)
            fake_inputs_b = torch.cat((photos, fake_sketches), 1)

            real_outputs_a = discriminator_a(real_inputs_a)
            fake_outputs_a = discriminator_a(fake_inputs_a)
            real_outputs_b = discriminator_b(real_inputs_b)
            fake_outputs_b = discriminator_b(fake_inputs_b)

            real_labels_a = torch.ones_like(real_outputs_a)
            fake_labels_a = torch.zeros_like(fake_outputs_a)
            real_labels_b = torch.ones_like(real_outputs_b)
            fake_labels_b = torch.zeros_like(fake_outputs_b)

            d_loss_real_a = criterion(real_outputs_a, real_labels_a)
            d_loss_fake_a = criterion(fake_outputs_a, fake_labels_a)
            d_loss_a = d_loss_real_a + d_loss_fake_a
            optimizer_d_a.zero_grad()
            d_loss_a.backward(retain_graph=True)
            optimizer_d_a.step()

            d_loss_real_b = criterion(real_outputs_b, real_labels_b)
            d_loss_fake_b = criterion(fake_outputs_b, fake_labels_b)
            d_loss_b = d_loss_real_b + d_loss_fake_b
            optimizer_d_b.zero_grad()
            d_loss_b.backward(retain_graph=True)
            optimizer_d_b.step()

            g_loss_ab = criterion(discriminator_b(fake_inputs_b), real_labels_b)
            g_loss_ba = criterion(discriminator_a(fake_inputs_a), real_labels_a)
            g_loss = g_loss_ab + g_loss_ba
            optimizer_g.zero_grad()
            g_loss.backward(retain_graph=True)  # Retain the graph here
            optimizer_g.step()
            print(i)

            if i % 100 == 0:
                save_image(fake_photos, f'output/fake_photo_epoch_{epoch}_batch_{i}.png')
                save_image(fake_sketches, f'output/fake_sketch_epoch_{epoch}_batch_{i}.png')

        torch.save(generator_ab.state_dict(), f'generator_ab_epoch_{epoch}.pth')
        torch.save(generator_ba.state_dict(), f'generator_ba_epoch_{epoch}.pth')
        torch.save(discriminator_a.state_dict(), f'discriminator_a_epoch_{epoch}.pth')
        torch.save(discriminator_b.state_dict(), f'discriminator_b_epoch_{epoch}.pth')

train_cyclegan()


