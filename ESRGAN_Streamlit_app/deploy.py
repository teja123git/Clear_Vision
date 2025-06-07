import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from io import BytesIO

# Modern UI CSS for dark background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main { 
        background-color: #1A202C; 
        color: #F7FAFC; 
        padding: 2rem 1rem; 
        font-family: 'Inter', sans-serif; 
        max-width: 1200px; 
        margin: 0 auto; 
        min-height: 100vh; 
    }
    h1 { 
        color: #F7FAFC; 
        font-weight: 700; 
        font-size: 2.25rem; 
        text-align: center; 
        margin-bottom: 1rem; 
    }
    .stFileUploader { 
        background-color: #2D3748; 
        border-radius: 8px; 
        padding: 1.5rem; 
        border: 1px solid #4A5568; 
        margin-bottom: 1.5rem; 
    }
    .stButton>button { 
        background-color: #4299E1; 
        color: #F7FAFC; 
        border-radius: 8px; 
        padding: 0.75rem 1.5rem; 
        font-weight: 500; 
        border: none; 
        transition: background-color 0.2s ease; 
    }
    .stButton>button:hover { 
        background-color: #3182CE; 
    }
    .stAlert { 
        background-color: #742A2A; 
        border-left: 4px solid #F56565; 
        border-radius: 8px; 
        padding: 1rem; 
        color: #FEE2E2; 
        font-size: 0.875rem; 
    }
    .image-caption { 
        font-size: 0.875rem; 
        color: #A0AEC0; 
        margin-top: 0.5rem; 
        font-weight: 500; 
        text-align: center; 
    }
    .metrics-card { 
        background-color: #2D3748; 
        border-radius: 8px; 
        padding: 1rem; 
        margin: 1.5rem 0; 
        font-size: 1rem; 
        color: #F7FAFC; 
        font-weight: 500; 
        border: 1px solid #4A5568; 
        text-align: center; 
    }
    .stSpinner { 
        text-align: center; 
        color: #4299E1; 
        font-weight: 500; 
    }
    .stSubheader { 
        color: #F7FAFC; 
        font-weight: 600; 
        font-size: 1.5rem; 
        margin-top: 2rem; 
    }
    @media (max-width: 768px) {
        h1 { font-size: 1.75rem; }
        .stFileUploader { padding: 1rem; }
        .metrics-card { padding: 0.75rem; font-size: 0.875rem; }
        .stSubheader { font-size: 1.25rem; }
    }
    </style>
""", unsafe_allow_html=True)

# ChannelAttention (from model.py)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

# RDB (from model.py)
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels + growth_rate * i, growth_rate, 3, padding=1)
            for i in range(num_layers)
        ])
        out_channels = in_channels + growth_rate * num_layers
        self.ca = ChannelAttention(out_channels)
        self.reduce = nn.Conv2d(out_channels, in_channels, 1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = torch.cat([out, layer(out)], dim=1)
        ca_weights = self.ca(out)
        attended_out = out * ca_weights
        reduced_out = self.reduce(attended_out)
        return reduced_out + x

# Generator (from model.py)
class Generator(nn.Module):
    def __init__(self, num_rrdb=6):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.rrdb = nn.Sequential(*[RDB(64, 32, 2) for _ in range(num_rrdb)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        initial = self.conv1(x)
        x = self.rrdb(initial)
        x = self.conv2(x) + initial
        x = self.upsample(x)
        return torch.tanh(self.conv3(x))

# Load model
@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(num_rrdb=6).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Checkpoint '{checkpoint_path}' not found. Please ensure it exists in the app directory.")
        return None, device
    except RuntimeError as e:
        st.error(f"Error loading model: {e}. Ensure 'generator_epoch_25.pth' matches the Generator architecture.")
        return None, device
    model.eval()
    return model, device

# Corrupt image (exact match with data_prep.py)
def corrupt_image(image):
    image_np = np.array(image)
    noise = np.random.normal(0, 10, image_np.shape).astype(np.float32)
    noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(noisy_image, (5, 5), 0.5)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, compressed_image = cv2.imencode('.jpg', blurred_image, encode_param)
    decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    return Image.fromarray(decompressed_image)

# Simulate training degradation (exact match with data_prep.py)
def simulate_degradation(image):
    # Input: [1, 3, 256, 256] tensor
    # Convert tensor to PIL Image
    image_pil = transforms.ToPILImage()(image.squeeze(0))
    transform_lr = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(corrupt_image),
        transforms.ToTensor(),
    ])
    degraded_tensor = transform_lr(image_pil).unsqueeze(0)
    return degraded_tensor

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # [0, 1]
    ])
    return transform(image).unsqueeze(0)  # [1, 3, 256, 256]

# Postprocess image
def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # [3, 1024, 1024]
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# Compute metrics (match Evaluation.py)
def compute_metrics(pred_img, gt_img):
    # Convert to NumPy arrays
    pred_np = np.array(pred_img).astype(np.uint8)
    gt_np = np.array(gt_img).astype(np.uint8)
    # Compute PSNR and SSIM
    psnr_value = peak_signal_noise_ratio(gt_np, pred_np)
    ssim_value = structural_similarity(gt_np, pred_np, channel_axis=2)
    return psnr_value, ssim_value

# Streamlit app
def main():
    st.title("ESRGAN Image Super-Resolution")
    st.markdown("""
        <p style='text-align: center; color: #A0AEC0; font-size: 1rem; margin-bottom: 1.5rem;'>
        Upload a 256x256 RGB image to enhance it to 1024x1024 using ESRGAN. View original, degraded, and super-resolved images with quality metrics.
        </p>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Select Image", type=["png", "jpg", "jpeg"], help="Upload a 256x256 RGB PNG or JPEG image")

    if uploaded_file is not None:
        # Load and validate image
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Invalid image format. Please upload a valid PNG or JPEG file.")
            return
        
        if image.mode != "RGB":
            st.error("Please upload an RGB image.")
            return

        # Preprocess
        input_tensor = preprocess_image(image)

        # Simulate degradation
        degraded_tensor = simulate_degradation(input_tensor)
        # Resize degraded for display only
        degraded_display = F.resize(degraded_tensor, (256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
        degraded_image = postprocess_image(degraded_display)

        # Load model
        checkpoint_path = "generator_epoch_25.pth"
        model, device = load_model(checkpoint_path)
        if model is None:
            return

        # Move input to device
        degraded_tensor = degraded_tensor.to(device)

        # Inference (direct, matching Evaluation.py)
        with st.spinner("Enhancing Image..."):
            with torch.no_grad():
                predicted_tensor = model(degraded_tensor)
            predicted_image = postprocess_image(predicted_tensor)

        # Compute metrics
        gt_image = image.resize((256, 256))  # Ensure same size for metrics
        psnr_value, ssim_value = compute_metrics(predicted_image.resize((256, 256)), gt_image)

        # Display comparison
        st.subheader("Results")
        cols = st.columns(3)
        with cols[0]:
            st.image(image, caption="Original (256x256)", use_column_width=True)
            st.markdown('<p class="image-caption">256x256</p>', unsafe_allow_html=True)
        with cols[1]:
            st.image(degraded_image, caption="Degraded (256x256)", use_column_width=True)
            st.markdown('<p class="image-caption">256x256</p>', unsafe_allow_html=True)
        with cols[2]:
            st.image(predicted_image, caption="Super-Resolved (1024x1024)", use_column_width=True)
            st.markdown('<p class="image-caption">1024x1024</p>', unsafe_allow_html=True)

        # Display metrics
        st.markdown(
            f'<div class="metrics-card">PSNR: <strong>{psnr_value:.2f} dB</strong> | SSIM: <strong>{ssim_value:.4f}</strong></div>',
            unsafe_allow_html=True
        )

        # Download button
        buf = BytesIO()
        predicted_image.save(buf, format="PNG")
        st.download_button(
            label="Download Super-Resolved Image",
            data=buf.getvalue(),
            file_name="esrgan_super_resolved.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()