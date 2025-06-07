import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torch.cuda.amp import GradScaler, autocast

# environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GAN Training
optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=50)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=50)
adversarial_loss = nn.BCEWithLogitsLoss().to(device)
scaler = GradScaler() 

def add_noise(images, std=0.01):
    noise = torch.randn_like(images) * std
    return images + noise.clamp(-0.1, 0.1)

epochs = 10
for epoch in range(epochs):
    generator.train()
    discriminator.train()
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        
        # Discriminator with mixed precision
        optimizer_D.zero_grad()
        with autocast():
            fake_imgs = generator(lr_imgs).detach()
            real_imgs_noisy = add_noise(hr_imgs)
            fake_imgs_noisy = add_noise(fake_imgs)
            real_logits = discriminator(real_imgs_noisy)
            fake_logits = discriminator(fake_imgs_noisy)
            real_labels = torch.full_like(real_logits, 0.95)
            fake_labels = torch.zeros_like(fake_logits)
            d_loss = (adversarial_loss(real_logits - fake_logits.mean(), real_labels) +
                      adversarial_loss(fake_logits - real_logits.mean(), fake_labels)) / 2
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Generator with mixed precision
        optimizer_G.zero_grad()
        with autocast():
            fake_imgs = generator(lr_imgs)
            fake_logits = discriminator(add_noise(fake_imgs))
            real_logits = discriminator(add_noise(hr_imgs))
            g_loss_adversarial = adversarial_loss(fake_logits - real_logits.mean(), torch.ones_like(fake_logits))
            g_loss_content = content_loss(vgg(normalize_for_vgg(fake_imgs)), vgg(normalize_for_vgg(hr_imgs)).detach())
            g_loss_pixel = pixel_loss(fake_imgs, hr_imgs)
            g_loss = g_loss_content + 5e-2 * g_loss_adversarial + 5e-3 * g_loss_pixel
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    scheduler_G.step()
    scheduler_D.step()
    torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")