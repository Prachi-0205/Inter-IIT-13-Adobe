import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import timm
import json
import torch.fft
import pywt
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import io
import glob


"""
This Script is Generating Result for Task 1
"""


# Configurations
FFT = False
IN_CHANS=3 if not FFT else 9
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
NUM_SAVE_STEPS=100
MODEL = "tf_efficientnet_b4"
CKPT_PATH = f"ckpts/{MODEL}_fft/epoch_3.pt" if FFT else f"ckpts/{MODEL}_100/epoch_3.pt"

#Defined Arguments
val_sets = [
        "perturbed_images_32"
    ]


# ------------------------------------------ Feature Extraction ------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    Extracts features using FFT and optional wavelet transform.

    Args:
        low_pass_radius (int): Radius for the low-pass filter in FFT.
        resolution (int): Resolution for resizing wavelet components.

    Methods:
        apply_low_pass_filter: Applies a low-pass filter to the FFT of an image.
        apply_high_pass_filter: Applies a high-pass filter to the FFT of an image.
        extract_fft_features: Extracts low-pass and high-pass features using FFT.
        extract_wavelet_features: Extracts wavelet features using the Haar wavelet transform.
        forward: Combines input image with FFT features.
    """
    def __init__(self, low_pass_radius=4,resolution=224):
        super(FeatureExtractor, self).__init__()
        self.low_pass_radius = low_pass_radius
        self.res = resolution

    def apply_low_pass_filter(self, fft_complex):
        """
        Applies a low-pass filter to the input FFT.

        Args:
            fft_complex (torch.Tensor): FFT-transformed image.

        Returns:
            torch.Tensor: Low-pass filtered FFT.
        """
        
        B, C, H, W = fft_complex.shape
        mask = torch.zeros(H, W)
        center_x, center_y = H // 2, W // 2

        for x in range(H):
            for y in range(W):
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) < self.low_pass_radius:
                    mask[x, y] = 1

        mask = mask.to(fft_complex.device).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(B, C, H, W)
        return fft_complex * mask

    def apply_high_pass_filter(self, fft_complex):
        """
        Applies a high-pass filter to the input FFT.

        Args:
            fft_complex (torch.Tensor): FFT-transformed image.

        Returns:
            torch.Tensor: High-pass filtered FFT.
        """
        B, C, H, W = fft_complex.shape
        mask = torch.ones(H, W)
        center_x, center_y = H // 2, W // 2

        for x in range(H):
            for y in range(W):
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) < self.low_pass_radius:
                    mask[x, y] = 0

        mask = mask.to(fft_complex.device).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(B, C, H, W)
        return fft_complex * mask

    def extract_fft_features(self, x):
        """
        Extracts FFT features (low-pass and high-pass).

        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Combined FFT features.
        """
        fft_complex = torch.fft.fft2(x)
        fft_complex_shifted = torch.fft.fftshift(fft_complex)

        
        low_pass = self.apply_low_pass_filter(fft_complex_shifted)
        high_pass = self.apply_high_pass_filter(fft_complex_shifted)

        
        low_pass_ifft = torch.fft.ifft2(torch.fft.ifftshift(low_pass)).abs()
        high_pass_ifft = torch.fft.ifft2(torch.fft.ifftshift(high_pass)).abs()

        
        low_pass_magnitude = torch.abs(low_pass_ifft)
        high_pass_magnitude = torch.abs(high_pass_ifft)

        
        low_pass_magnitude = low_pass_magnitude / torch.max(low_pass_magnitude)
        high_pass_magnitude = high_pass_magnitude / torch.max(high_pass_magnitude)

        
        fft_features = torch.cat([low_pass_magnitude, high_pass_magnitude], dim=1)  # Shape: (B, 6, 32, 32)
        return fft_features

    def extract_wavelet_features(self, x):
        """
        Extracts wavelet features using the Haar wavelet transform.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted wavelet features of shape (batch_size, channels * 4, self.res, self.res).
        """
        batch_size, channels, height, width = x.size()
        wavelet_features = []

        for b in range(batch_size):
            for c in range(channels):
                img = x[b, c].cpu().detach().numpy()
                coeffs = pywt.dwt2(img, 'haar')
                cA, (cH, cV, cD) = coeffs

                # Resize components
                cA = torch.tensor(cA).float()
                cH = torch.tensor(cH).float()
                cV = torch.tensor(cV).float()
                cD = torch.tensor(cD).float()

                cA = nn.functional.interpolate(cA.unsqueeze(0).unsqueeze(0), size=(self.res,self.res), mode='bilinear').squeeze()
                cH = nn.functional.interpolate(cH.unsqueeze(0).unsqueeze(0), size=(self.res,self.res), mode='bilinear').squeeze()
                cV = nn.functional.interpolate(cV.unsqueeze(0).unsqueeze(0), size=(self.res,self.res), mode='bilinear').squeeze()
                cD = nn.functional.interpolate(cD.unsqueeze(0).unsqueeze(0), size=(self.res,self.res), mode='bilinear').squeeze()

                wavelet_features.extend([cA, cH, cV, cD])

        wavelet_features = torch.stack(wavelet_features).view(batch_size, channels * 4, height, width)
        return wavelet_features

    def forward(self, x):
        """
        Combines input image with FFT features.

        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Image combined with FFT features.
        """
        fft_features = self.extract_fft_features(x)       
        # wavelet_features = self.extract_wavelet_features(x)  
        combined_features = torch.cat((x, fft_features.to(x.device,x.dtype)), dim=1)
        return combined_features
    


class Classifier(nn.Module):
    """
    A classifier model with a feature extractor backbone.

    Args:
        model (str): Model name (e.g., EfficientNet, ResNet).
        subnets (bool): Whether to include sub-networks.

    Methods:
        forward: Performs a forward pass through the model.
    """
    def __init__(self,model,subnets=False):
        super().__init__()
        self.bb = timm.create_model(model,pretrained=False,in_chans=IN_CHANS,features_only=True)
        self.subnets = subnets
        if model.startswith("resnet50") or model.startswith("resnetv2_50"):
            self.head = nn.Sequential(
                nn.Conv2d(2048,32,3,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32*9,32),
                nn.ReLU(),
                nn.Linear(32,1),

            )
        elif model.startswith("mobilenetv4_hybrid_large_075"):
            self.head = nn.Sequential(
                nn.Conv2d(720,32,3,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32*9,32),
                nn.ReLU(),
                nn.Linear(32,1),

            )
        elif model.startswith("swinv2_cr_tiny_ns_224"):
            self.head = nn.Sequential(
                nn.Conv2d(768,32,3,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32*9,32),
                nn.ReLU(),
                nn.Linear(32,1),

            )
        elif model.startswith("tf_efficientnet_b4"):
            self.head = nn.Sequential(
                nn.Conv2d(448,128,3,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*9,32),
                nn.ReLU(),
                nn.Linear(32,1),
            )
        elif model.startswith("tf_efficientnet_b3"):
            self.head = nn.Sequential(
                nn.Conv2d(384,128,3,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*9,32),
                nn.ReLU(),
                nn.Linear(32,1),

            )
        if subnets and model.startswith("tf_efficientnet_b3"):
            self.subhead1 = nn.Sequential(
                nn.Conv2d(24, 96, 3, 2,1),  
                nn.ReLU(),
                nn.GroupNorm(4, 96),
                
                
                nn.Conv2d(96, 192, 3, 2,1),  
                nn.ReLU(),
                nn.GroupNorm(4, 192),
                
                
                nn.Conv2d(192, 192, 3, 2,1),  
                nn.ReLU(),
                nn.GroupNorm(4, 192),
                
                
                nn.Conv2d(192, 192, 3, 2,1),  
                nn.ReLU(),
                nn.GroupNorm(4, 192),
                
            )
            
            self.subhead2 = nn.Sequential(
                nn.Conv2d(48, 192, 3, 2,1),
                nn.ReLU(),
                nn.GroupNorm(4, 192),
                
                nn.Conv2d(192, 192, 3, 2,1),
                nn.ReLU(),
                nn.GroupNorm(4, 192),
            )
            self.supersubhead=nn.Sequential(
                nn.Conv2d(768,32,3,3,1),
                nn.ReLU(),
                nn.GroupNorm(4,32),
                nn.Flatten(),
                nn.Linear(32*3*3,1)
            )
            
        self.bb.requires_grad_ = True
        self.bb.train()
        self.head.requires_grad_ = True
        if subnets and hasattr(self, 'subhead1'):
            self.subnets = True
            self.subhead1.requires_grad_=True
            self.subhead2.requires_grad_=True
    
    def forward(self, x, temperature=1):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            temperature (float, optional): Scaling factor for the output. Default is 1.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - Main classification output if subnets=False.
                - Tuple of main output and sub-network output if subnets=True.
        """
        features = self.bb(x)
        out = F.sigmoid(self.head(features[-1])/temperature)
        
        if self.subnets and hasattr(self, 'subhead1'):
            s1 = self.subhead1(features[0])
            s2 = self.subhead2(features[2])
            concat_data = torch.cat([s1, s2, features[-1]], dim=1)
            s3 = self.supersubhead(concat_data)
            return out, F.sigmoid(s3/temperature)
        else:
            return out


class TestDataset(Dataset):
    """
    Custom Dataset class for loading images from a directory and applying transformations.

    Args:
        data_path (str): Path to the main dataset directory.
        transform (callable, optional): Transformations to apply to the images.

    Attributes:
        data (list): List of image file paths.
        transform (callable): Transformations to apply to the images.
    """

    def __init__(self, data_path, transform=None):

        """
        Args:
            primary_dataset_path (str): Path to the main dataset directory.
            additional_dirs (list): List of paths to additional directories.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.transform = transform
        self.root_dir = data_path
        data_paths = glob.glob('**/*.png', recursive=True,root_dir=self.root_dir) + glob.glob('**/*.jpg', recursive=True,root_dir=self.root_dir)

        self.data = []

        for data_item in data_paths:
            is_png = data_item.lower().endswith("png")
            self.data.append((self.root_dir+"/"+data_item))

        print(len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and transform an image by its index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Transformed image tensor and the corresponding file path.
        """
        img_path = self.data[idx]
        image = Image.open(img_path).convert("RGB").resize((32,32),Image.BILINEAR)

        # if is_png:
        #     with io.BytesIO() as buffer:
        #         image.save(buffer, format="JPEG")
        #         buffer.seek(0)
        #         image = Image.open(buffer)
        #         image.load()

        if self.transform:
            image = self.transform(image)

        return image,img_path

#---------------------------------------------------Initialisation-----------------------------------------------

data_transforms = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Initialize model
model = Classifier(MODEL,subnets=False).to("cuda")
model.eval()
model.requires_grad_(False)


fft_extractor = FeatureExtractor(2,224)
model.load_state_dict(torch.load(CKPT_PATH,weights_only=True))

#---------------------------------------------------------- Inference --------------------------------------------
results= []
num_fake = 0
num_real = 0
for val_set in val_sets:

    with torch.no_grad():
        path= val_set
        dataset = TestDataset(path, transform=data_transforms)
        dataloader = DataLoader(dataset,batch_size=256,num_workers=8)

        for batch in dataloader:
            images, img_paths = batch

            if FFT:
                imgs = fft_extractor(images.to("cuda"))
            else:
                imgs = images.to("cuda")

            if model.subnets:
                preds,preds_subs = model(imgs)
            else:
                preds = model(imgs)


            preds = (preds>0.5).float()
            for prediction,path in zip(preds,img_paths):

                if prediction.item()==1.0:
                    add_value = "fake"
                    num_fake+=1
                else:
                    add_value = "real"
                    num_real+=1
                path = path.split("/")[-1].split(".")[0]
                results.append({"index":path,"prediction":add_value})


with open("22_task1.json","w") as f:
    json.dump(results,f)
print(num_fake,num_real)
            



        
 