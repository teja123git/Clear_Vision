import torch
from torchvision.models import vgg19
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(num_rrdb=6).to(device)
discriminator = Discriminator().to(device)
vgg = vgg19(weights='VGG19_Weights.DEFAULT').features[:36].eval().to(device)  
for param in vgg.parameters():
    param.requires_grad = False

vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def normalize_for_vgg(img):
    return vgg_normalize((img + 1) / 2)

content_loss = nn.MSELoss().to(device)
pixel_loss = nn.L1Loss().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4)

print("Pre-training generator...")
epochs = 5
for epoch in range(epochs):
    generator.train()
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        
        optimizer_G.zero_grad()
        fake_imgs = generator(lr_imgs)
  
        with torch.no_grad():
            vgg_fake = vgg(normalize_for_vgg(fake_imgs))
            vgg_hr = vgg(normalize_for_vgg(hr_imgs))
        
        min_h = min(vgg_fake.shape[2], vgg_hr.shape[2])
        min_w = min(vgg_fake.shape[3], vgg_hr.shape[3])
        vgg_fake = F.interpolate(vgg_fake, size=(min_h, min_w), mode='bilinear', align_corners=False)
        vgg_hr = F.interpolate(vgg_hr, size=(min_h, min_w), mode='bilinear', align_corners=False)
        
        g_loss_perceptual = content_loss(vgg_fake, vgg_hr.detach())
        g_loss_pixel = pixel_loss(fake_imgs, hr_imgs)
        g_loss = g_loss_perceptual + 0.01 * g_loss_pixel
        g_loss.backward()
        optimizer_G.step()
    print(f"Pre-train Epoch [{epoch+1}/{epochs}], G Loss: {g_loss.item():.4f}")