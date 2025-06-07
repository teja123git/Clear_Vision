import numpy as np
# from torchmetrics.image import FrechetInceptionDistance
# from torchmetrics.functional import learned_perceptual_image_patch_similarity as lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def evaluate_model(generator, test_loader):
    generator.eval()
    # fid = FrechetInceptionDistance(feature=2048).to(device)
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    # fid.reset()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            fake_imgs = generator(lr_imgs)
            vutils.save_image(fake_imgs, f"output_epoch_{i}.png", normalize=True)
            fake_np = fake_imgs.cpu().numpy().transpose(0, 2, 3, 1)
            hr_np = hr_imgs.cpu().numpy().transpose(0, 2, 3, 1)
            for fake, hr in zip(fake_np, hr_np):
                fake = (fake * 255).astype(np.uint8)
                hr = (hr * 255).astype(np.uint8)
                psnr_scores.append(peak_signal_noise_ratio(hr, fake))
                ssim_scores.append(structural_similarity(hr, fake, channel_axis=2))
            # lpips_scores.append(lpips(fake_imgs, hr_imgs, normalize=True).mean().item())
            # fid.update((hr_imgs * 255).to(torch.uint8), real=True)
            # fid.update((fake_imgs * 255).to(torch.uint8), real=False)
            if i == 0:
                plt.figure(figsize=(12, 9))
                for j in range(4):
                    plt.subplot(3, 4, j + 1)
                    plt.imshow(lr_imgs[j].cpu().permute(1, 2, 0).numpy())
                    plt.title("Low Res")
                    plt.axis("off")
                    plt.subplot(3, 4, j + 5)
                    plt.imshow(fake_imgs[j].cpu().permute(1, 2, 0).numpy())
                    plt.title("Restored")
                    plt.axis("off")
                    plt.subplot(3, 4, j + 9)
                    plt.imshow(hr_imgs[j].cpu().permute(1, 2, 0).numpy())
                    plt.title("High Res")
                    plt.axis("off")
                plt.show()
            if i == 10:
                break
    print(f"Average PSNR: {np.mean(psnr_scores):.2f}, SSIM: {np.mean(ssim_scores):.4f}, "
          f"LPIPS: {np.mean(lpips_scores):.4f}")
    # print(f"FID: {fid.compute().item():.2f}")

evaluate_model(generator, test_loader)