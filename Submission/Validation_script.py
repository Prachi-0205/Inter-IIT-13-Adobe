import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import timm
import numpy as np
import torch.fft
import pywt
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import io
import glob

FFT = True
IN_CHANS=3 if not FFT else 9
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
NUM_SAVE_STEPS=100
MODEL = "coatnet_nano_rw_224"  #Provide the Model Name
CKPT_PATH="/workspace/ckpts/coatnet_nano_rw_224_fft_100_GAN/epoch_4.pt"  # Provide PAth_to_Checkpoints Saved accoring to the scripts

val_sets = [
    ("REAL",0),     # Additional validation dataset can be provided generated according to the script image generation script
    ("FAKE",1),
    
    ]


class FeatureExtractor(nn.Module):
    """
    Feature extraction using FFT and Wavelet transforms.

    Attributes
    ----------
    low_pass_radius : int (Radius for low-pass filter in FFT.)
    resolution : int (Resolution to which the features will be interpolated.)

    Methods
    -------
    apply_low_pass_filter(fft_complex)
        Applies a low-pass filter to the FFT-transformed input.

    apply_high_pass_filter(fft_complex)
        Applies a high-pass filter to the FFT-transformed input.

    extract_fft_features(x)
        Extracts FFT features using low-pass and high-pass filters.

    extract_wavelet_features(x)
        Extracts Wavelet features from the input tensor.

    forward(x)
        Combines FFT and Wavelet features with the input tensor.
    """
    def __init__(self, low_pass_radius=4,resolution=224):
        super(FeatureExtractor, self).__init__()
        self.low_pass_radius = low_pass_radius
        self.res = resolution

    def apply_low_pass_filter(self, fft_complex):
        """
        Apply a low-pass filter to the FFT-transformed input.

        Parameters
        ----------
        fft_complex : torch.Tensor
            Complex tensor of FFT-transformed input of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Low-pass filtered tensor of shape (B, C, H, W).
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
        Apply a high-pass filter to the FFT-transformed input.

        Parameters
        ----------
        fft_complex : torch.Tensor
            Complex tensor of FFT-transformed input of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            High-pass filtered tensor of shape (B, C, H, W).
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
        Extract FFT features using low-pass and high-pass filters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            FFT features tensor of shape (B, 6, H, W).
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
        Extract Wavelet features using Discrete Wavelet Transform.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Wavelet features tensor of shape (B, C*4, resolution, resolution).
        """
        batch_size, channels, height, width = x.size()
        wavelet_features = []

        for b in range(batch_size):
            for c in range(channels):
                img = x[b, c].cpu().detach().numpy()
                coeffs = pywt.dwt2(img, 'haar')
                cA, (cH, cV, cD) = coeffs

                
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
        Perform a forward pass to combine original, and  FFT features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Combined features tensor of shape (B, C+6, H, W).
        """
        fft_features = self.extract_fft_features(x)       
        # wavelet_features = self.extract_wavelet_features(x)  
        combined_features = torch.cat((x, fft_features.to(x.device,x.dtype)), dim=1)
        return combined_features
    


class Classifier(nn.Module):
    """
    Neural network classifier with backbone and custom head.

    Attributes
    ----------
    bb : timm.models.features_only
        Backbone feature extractor from timm.
    head : nn.Sequential
        Custom head for classification.

    Methods
    -------
    forward(x, temperature=1)
        Perform a forward pass and return predictions.
    """
    def __init__(self,model,subnets=False):
        """
        Initialize the classifier with a backbone model.

        Parameters
        ----------
        model : str
            Model name from timm library.
        """
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
        else:
            self.head = nn.Sequential(
                nn.Conv2d(512,128,3,2),
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
        Perform a forward pass and return predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        temperature : float, optional
            Temperature scaling for the logits (default is 1).

        Returns
        -------
        torch.Tensor
            Sigmoid-activated predictions of shape (B, 1).
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



class ValDataset(Dataset):
    def __init__(self, data_path, label, transform=None):
        """
        Args:
            primary_dataset_path (str): Path to the main dataset directory.
            additional_dirs (list): List of paths to additional directories.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.transform = transform
        self.root_dir = data_path
        self.label = label
        data_paths = glob.glob('**/*.png', recursive=True,root_dir=self.root_dir) + glob.glob('**/*.jpg', recursive=True,root_dir=self.root_dir)

        self.data = []

        for data_item in data_paths:
            is_png = data_item.lower().endswith("png")
            self.data.append((self.root_dir+"/"+data_item,label,is_png))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, is_png = self.data[idx]
        image = Image.open(img_path).convert("RGB").resize((32,32),Image.BILINEAR)

        if is_png:
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                buffer.seek(0)
                image = Image.open(buffer)
                image.load()

        if self.transform:
            image = self.transform(image)

        return image, label

#--------------------------------------------Data Transformation and Model ---------------------------------------------------

data_transforms = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

model = Classifier(MODEL,subnets=False).to("cuda")
model.eval()
model.requires_grad_(False)
fft_extractor = FeatureExtractor(2,224)
model.load_state_dict(torch.load(CKPT_PATH,weights_only=True))

T_tp = 0
T_tn = 0 
T_fp = 0
T_fn = 0

S_T_tp = 0
S_T_tn = 0
S_T_fp = 0
S_T_fn = 0

#---------------------------------------------------Validation---------------------------------------------------

for val_set in val_sets:

    with torch.no_grad():
        path, label = val_set
        dataset = ValDataset(path, label, transform=data_transforms)
        dataloader = DataLoader(dataset,batch_size=256,num_workers=8)

        tp = 0
        tn = 0 
        fp = 0
        fn = 0

        s_tp =0
        s_tn = 0
        s_fp = 0
        s_fn = 0

        for batch in dataloader:
            images, labels = batch

            bsz = labels.shape[0]
            if FFT:
                imgs = fft_extractor(images.to("cuda"))
            else:
                imgs = images.to("cuda")

            if model.subnets:
                preds,preds_subs = model(imgs)
            else:
                preds = model(imgs)

            labels = labels.float().unsqueeze(-1).to("cuda")

            preds = (preds>0.5).float()
            if model.subnets:
                preds_subs = (preds_subs>0.5).float()

            true_positives = ((preds == 1) & (labels == 1)).float().sum()
            false_positives = ((preds == 1) & (labels == 0)).float().sum()
            true_negatives = ((preds == 0) & (labels == 0)).float().sum()
            false_negatives = ((preds == 0) & (labels == 1)).float().sum()


            if model.subnets:


                s_true_positives = ((preds_subs == 1) & (labels == 1)).float().sum()
                s_false_positives = ((preds_subs == 1) & (labels == 0)).float().sum()
                s_true_negatives = ((preds_subs == 0) & (labels == 0)).float().sum()
                s_false_negatives = ((preds_subs == 0) & (labels == 1)).float().sum()


            tp+=true_positives
            tn+=true_negatives
            fp+=false_positives
            fn+=false_negatives
    
            if model.subnets:
                s_tp+=s_true_positives
                s_tn+=s_true_negatives
                s_fp+=s_false_positives
                s_fn+=s_false_negatives


    T_tp +=tp
    T_tn +=tn
    T_fp += fp
    T_fn +=fn

    S_T_tp +=s_tp
    S_T_tn +=s_tn
    S_T_fp += s_fp
    S_T_fn +=s_fn



            
    print(f"Validation on {val_set[0]}")
    print(f"""{torch.tensor([
        [tp, fp],
        [fn, tn]
    ])}""")
    print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn)}")


    if model.subnets:
        print(f"Subnetwork Validation on {val_set[0]}")
        print(f"""{torch.tensor([
            [s_tp, s_fp],
            [s_fn, s_tn]
        ])}""")
        print(f"Accuracy: {(s_tp+s_tn)/(s_tp+s_tn+s_fp+s_fn)}")

print(f"Validation Total")
print(f"""{torch.tensor([
    [T_tp, T_fp],
    [T_fn, T_tn]
])}""")
print(f"Accuracy: {(T_tp+T_tn)/(T_tp+T_tn+T_fp+T_fn)}")


if model.subnets:
    print(f"Validation Total")
    print(f"""{torch.tensor([
        [S_T_tp, S_T_fp],
        [S_T_fn, S_T_tn]
    ])}""")
    print(f"Accuracy: {(S_T_tp+S_T_tn)/(S_T_tp+S_T_tn+S_T_fp+S_T_fn)}")