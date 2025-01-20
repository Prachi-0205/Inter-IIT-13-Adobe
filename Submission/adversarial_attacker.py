"""
This python file generate different adversarial attacks for the model
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import timm
import torch.fft
import pywt
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import io
import glob
import random
import torch.optim as optim
import time
from typing import List, Optional, Union
import tracemalloc
import psutil
import tracemalloc
import logging
from tqdm import tqdm





FFT = False # do not touch if model is not FFT based
IN_CHANS=3 if not FFT else 9 
RESIZE_HEIGHT = 224 # size input size for models
RESIZE_WIDTH = 224
NUM_SAVE_STEPS=100 
TRAIN_TEMP=10
MODEL = "tf_efficientnet_b4" # put model name here check classifier class for help
CKPT_PATH = f"../ckpts/tf_efficientnet_b4_120_GAN/epoch_3.pt" # put model checkpoints paths

ATTACK_NAME="DeepFool"  # enter the attack name from this list [DeepFool, HopSkipJump, C&W, Spatial, PGD]
MEAN=(0.5,0.5,0.5)
STD=(0.5,0.5,0.5)

DATA_LIM=32# How many images to load
LENGTH=4 # How many images to take inference on should be <=DATA_LIM
BATCH_SIZE=4
STORING_DIR="IMAGES"  # make sure there is a REAL and FAKE sub-dir already created in this folder
DATASET_PATH="../archive/test"

val_sets = [
# Put validation data paths here, in a tuple with the label for the folder
    ]



failed_num=0
img_id=0


class FeatureExtractor(nn.Module):
    """
    A feature extractor module that supports FFT-based and wavelet-based feature extraction.
    """
    def __init__(self, low_pass_radius=4,resolution=224):
        super(FeatureExtractor, self).__init__()
        self.low_pass_radius = low_pass_radius
        self.res = resolution

    def apply_low_pass_filter(self, fft_complex):
        # Create a low-pass mask
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
        # Create a high-pass mask
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
        fft_features = self.extract_fft_features(x)       
       
        combined_features = torch.cat((x, fft_features.to(x.device,x.dtype)), dim=1)
        return combined_features
    
class Classifier(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.bb = timm.create_model(model,pretrained=False,in_chans=IN_CHANS,features_only=True)
        if model.startswith("resnet50") or model.startswith("resnetv2_50") or model.startswith("resnext50_32x4d"):
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
    def forward(self,x,temperature = 1):
        features = self.bb(x)

        out = F.sigmoid(self.head(features[-1])/temperature)
        return out


class ValDataset(Dataset):
    def __init__(self, data_path, label, transform=None):
       
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



data_transforms = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN,STD)
])
transform=transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN,STD)
])
preprocess = transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(MODEL).to("cuda")
model.eval()
model.load_state_dict(torch.load(CKPT_PATH,weights_only=True))
fft_extractor = FeatureExtractor(resolution=224,low_pass_radius=4)



def get_inference(image_path,model=model):
    imj=[]
    image = Image.open(image_path).convert("RGB")
    image_array=transform(image)
    imj.append(image_array)
    images_ = torch.stack(imj)
    if isinstance(images_, list):
            images_ = torch.tensor(images_, dtype=torch.float32)
    if isinstance(images_, np.ndarray):
            images_ = torch.tensor(images_, dtype=torch.float32).permute(0, 3, 1, 2)        
    images_=images_.to(device)
    model.eval()
    outputs=model(images_)
    return outputs



def denormalize_image_vit(image_tensor):

    device = image_tensor.device
    mean = torch.tensor(MEAN, device=device)
    std = torch.tensor(STD, device=device)
    if FFT:
        image_tensor = image_tensor[:3,:,:]

    image_tensor = image_tensor.permute(1, 2, 0)
    denormalized = (image_tensor * std + mean).clip(0, 1) 
    denormalized = (denormalized * 255).round() 

    return denormalized.detach().cpu().numpy().astype(np.uint8)


def load_images_from_path(path, label_map, image_size=(224, 224), batch_size=16):
   
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), 
        transforms.Normalize(MEAN,STD)
    ])
    images = []
    labels = []
    val = 0

    for label_name, class_index in label_map.items():
        class_path = os.path.join(path, label_name)
        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(class_path, filename)
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image)  
                images.append(image_tensor)
                labels.append(class_index)
                val += 1
                if val > DATA_LIM:
                    break
        if val > DATA_LIM:
            break  


    images = torch.stack(images)  
    labels = torch.tensor(labels) 


    num_batches = len(images) // batch_size
    batched_images = [images[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    batched_labels = [labels[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    return batched_images, batched_labels


def pgd_attack_batch(model, images, labels, epsilon=0.03, alpha=0.01, iterations=10):
  
    model.eval()
    if isinstance(images, list):
        images = torch.tensor(images, dtype=torch.float32)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.long)

    images = images.to(device)
    labels = labels.to(device).to(torch.float32)
    adversarial_images = images.clone().detach().requires_grad_(True)
    for _ in range(iterations):

        outputs = model(adversarial_images) 
        print(outputs, "outputs----------------------------------------------------",iterations,_)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.squeeze(), labels)

        model.zero_grad()
        loss.backward()
        gradient_sign = adversarial_images.grad.data.sign()
        adversarial_images = adversarial_images + alpha * gradient_sign
        perturbation = torch.clamp(adversarial_images - images, -epsilon, epsilon)
        adversarial_images = torch.clamp(images + perturbation, 0, 1).detach().requires_grad_(True)

    return adversarial_images
    

    


def test_pgd_attack_batch(model, test_images, test_labels, batch_size=16, epsilon=0.03, alpha=0.01, iterations=10, device='cuda'):

    if isinstance(test_images, np.ndarray):
        test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)
    if isinstance(test_labels, list):
        test_labels = torch.tensor(test_labels, dtype=torch.long)
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.tensor(test_labels, dtype=torch.long)    

    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    model.to(device)


    num_batches = len(test_images) // batch_size

    global img_id  

    for i in range(num_batches):
  
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        if FFT:
                batch_images = fft_extractor(batch_images.to("cuda"))
                print(batch_images.shape)
        else:
            batch_images = batch_images.to("cuda")
        adversarial_images = pgd_attack_batch(model, batch_images, batch_labels, epsilon, alpha, iterations)
        original_logits = model(batch_images) # Correctly pass pixel_values
        adversarial_logits = model(adversarial_images)
                
        original_preds = original_logits>0.5
        adversarial_preds = adversarial_logits>0.5

        print(f"Batch {i + 1}:")
        for j in range(batch_size):
            print(f"Image {j + 1}: Original Prediction: {original_logits[j].item()}, Adversarial Prediction: {adversarial_logits[j].item()}")
            print(adversarial_images[j].shape)
            adversarial_image_np = denormalize_image_vit(adversarial_images[j])  # Denormalize one adversarial image
            adversarial_image_np1 = Image.fromarray(adversarial_image_np)
            adversarial_image_np1_resized = adversarial_image_np1.resize((32, 32))

            image_real_np=denormalize_image_vit(batch_images[j])
            real_image_np1 = Image.fromarray(image_real_np)
            real_image_np1_resized = real_image_np1.resize((32, 32))

            str = "REAL"
            if batch_labels[j] == 1:
                str = "FAKE"
            
            adversarial_image_np1_resized.save(f'{STORING_DIR}/{str}/{img_id}.png')
            real_image_np1_resized.save(f'{STORING_DIR}/{str}/{img_id}_original.png')
            img_id += 1
            if original_preds[j] != adversarial_preds[j]:
                print("  Success: Adversarial image fooled the model!")
                
            else:
                print("  Failure: Adversarial image did not fool the model. Try increasing epsilon.")
                global failed_num
                failed_num=failed_num+1



def get_prediction(model_output):
    
    return (model_output >= 0.5).long()


    
def batched_spatial_transformation_attack(image_batch, model, max_rotation=100, max_translation=1):
    model.eval()
    device = next(model.parameters()).device
    image_batch = image_batch.to(device)
    
    # Get the original predictions
    with torch.no_grad():
        orig_preds = model(image_batch)
        orig_labels = get_prediction(orig_preds)

    batch_size = image_batch.size(0)
    perturbed_images = image_batch.clone()  # To store the adversarial images
    
    for _ in range(100):  # Number of iterations for random transformations
        for i in range(batch_size):
            # Random transformation parameters
            rot = random.uniform(-max_rotation, max_rotation)
            tx = random.uniform(-max_translation, max_translation)
            ty = random.uniform(-max_translation, max_translation)
            
            # Construct transformation matrix
            theta = torch.tensor([
                [np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), tx],
                [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), ty]
            ], dtype=torch.float, device=device).unsqueeze(0)
            
            # Generate grid and transform image
            grid = F.affine_grid(theta, image_batch[i:i+1].size(), align_corners=False)
            transformed_image = F.grid_sample(image_batch[i:i+1], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            # print(transformed_image==image_batch[i:i+1],"lop")
            
            # Check model prediction on transformed image
            with torch.no_grad():
                pred = model(transformed_image)
                pred2=model(image_batch[i:i+1])
                pred_label = get_prediction(pred)
                print(get_prediction(pred),get_prediction(pred2))
                
            
            # Update perturbed image if it changes the prediction
            if pred_label != orig_labels[i]:
                print("here")
                perturbed_images[i] = transformed_image.squeeze(0)
    
    return perturbed_images, orig_labels


class ImageFolderDataset(Dataset):
    """
    Custom Dataset for loading images from a directory
    
    Args:
        root_dir (str): Directory with all the images
        transform (callable, optional): Optional transform to be applied on a sample
        label_mapper (callable, optional): Function to map image path to label
    """
    def __init__(self, root_dir, transform=None, label_mapper=None, recursive=True):
        self.image_paths = []
        self.labels = []
        
        # Support recursive and non-recursive image collection
        if recursive:
            for root, _, files in os.walk(root_dir):
                for filename in files:
                    if self._is_image_file(filename):
                        full_path = os.path.join(root, filename)
                        self.image_paths.append(full_path)
                        
                        # Use label mapper if provided, otherwise default to parent folder name
                        if label_mapper:
                            self.labels.append(label_mapper(full_path))
                        else:
                            self.labels.append(os.path.basename(os.path.dirname(full_path)))
        else:
            for filename in os.listdir(root_dir):
                full_path = os.path.join(root_dir, filename)
                if self._is_image_file(filename):
                    self.image_paths.append(full_path)
                    
                    # Use label mapper if provided, otherwise default to parent folder name
                    if label_mapper:
                        self.labels.append(label_mapper(full_path))
                    else:
                        self.labels.append(os.path.basename(os.path.dirname(full_path)))
        
        if not self.image_paths:
            raise ValueError(f"No image files found in {root_dir}. Check the directory path.")
        
        self.transform = transform

    def _is_image_file(self, filename):
        """Check if a file is an image based on its extension."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def generate_adversarial_images(
    dataset_path, 
    model, 
    num_images=10, 
    batch_size=4, 
    output_dir='adversarial_images', 
    max_rotation=5, 
    max_translation=0.02,
    recursive=True
):
    
  
    os.makedirs(output_dir, exist_ok=True)
    preprocess_input = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
        
    ])
    final_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    #  adjust accordingly
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    

    dataset = ImageFolderDataset(dataset_path, transform=preprocess_input, recursive=recursive)
    batch_size = min(batch_size, len(dataset))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    total_generated = 0
    
    
    for batch, (images, labels, paths) in enumerate(dataloader):
        
        if total_generated >= num_images:
            break
        
        
        adversarial_batch, orig_labels = batched_spatial_transformation_attack(
            images, 
            model, 
            max_rotation=max_rotation, 
            max_translation=max_translation
        )
        
       
        for i, (adv_img, orig_label, orig_path) in enumerate(zip(adversarial_batch, orig_labels, paths)):
            if total_generated >= num_images:
                break
            
            orig_image=preprocess_input(Image.open(orig_path).convert("RGB")).to(device)      
            orig_pred=model(orig_image.unsqueeze(0))[0][0]>0.5
            adv_pred=model(adv_img.unsqueeze(0))[0][0]>0.5

            print(orig_pred,adv_pred)
            adv_img_processed = denormalize(adv_img).clamp(0, 1)
            

            pil_img = transforms.ToPILImage()(adv_img_processed)
            pil_img_32 = pil_img.resize((32, 32), Image.Resampling.LANCZOS)

            label_name = dataset.labels[dataset.image_paths.index(orig_path)]            
            label_output_dir = os.path.join(output_dir, label_name)
            os.makedirs(label_output_dir, exist_ok=True)
            
            filename = f'adversarial_{total_generated}_{os.path.basename(orig_path)}'
            save_path = os.path.join(label_output_dir, filename)
            pil_img_32.save(save_path)
            adv_img_processed = denormalize(orig_image).clamp(0, 1)
            
            pil_img = transforms.ToPILImage()(adv_img_processed)
            pil_img_32 = pil_img.resize((32, 32), Image.Resampling.LANCZOS)
            
            label_name = dataset.labels[dataset.image_paths.index(orig_path)]
        
            label_output_dir = os.path.join(output_dir, label_name)
            os.makedirs(label_output_dir, exist_ok=True)
            
        
            filename = f'Real_{total_generated}_{os.path.basename(orig_path)}'
            save_path = os.path.join(label_output_dir, filename)
            pil_img_32.save(save_path)
            
            total_generated += 1
            
            print(f"Generated adversarial image {total_generated}: {filename} (Label: {label_name})")
            if(orig_pred!=adv_pred):
                print("succefully fooled the model")
                global failed_num
                failed_num+=1
            else:
                print("Failed to fool the model")
    
    print(f"Finished generating {total_generated} adversarial images. The fooling rate is {failed_num/32}")

def measure_performance(func):
    """
    Decorator to measure function performance
    """
    def wrapper(*args, **kwargs):
        
        tracemalloc.start()
        
       
        start_time = time.time()
        start_cpu = time.process_time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        
        result = func(*args, **kwargs)
        
        
        end_time = time.time()
        end_cpu = time.process_time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Print performance metrics
        print("\n--- Performance Metrics ---")
        print(f"Wall Clock Time: {end_time - start_time:.4f} seconds")
        print(f"CPU Processing Time: {end_cpu - start_cpu:.4f} seconds")
        print(f"Memory Usage:")
        print(f"  Current Memory: {current / 1024:.2f} MB")
        print(f"  Peak Memory: {peak / 1024:.2f} MB")
        print(f"  Actual Memory Change: {end_memory - start_memory:.2f} MB")
        
        return result
    return wrapper

class CarliniWagnerAttack:
    def __init__(
        self, 
        model, 
        device: Optional[torch.device] = None, 
        confidence: float = 0, 
        learning_rate: float = 0.01,
        binary_search_steps: int = 9, 
        max_iterations: int = 10000,
        initial_const: float = 1e-3,
        abort_early: bool = True
    ):
        """
        Initialize Carlini-Wagner Attack
        
        Args:
            model: Classification model to attack
            device: Computation device (cuda/cpu)
            confidence: Confidence parameter for C&W attack
            learning_rate: Learning rate for optimization
            binary_search_steps: Number of binary search steps
            max_iterations: Maximum optimization iterations
            initial_const: Initial constant for the optimization
            abort_early: Whether to abort optimization early
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # C&W attack hyperparameters
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.abort_early = abort_early

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Image preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess an image for the model
        """
        if isinstance(image, (np.ndarray, Image.Image)):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.preprocess(image).unsqueeze(0)
        
        return image.to(self.device)

    def _get_prediction(self, img):
        """
        Get model prediction with robust handling
        """
        with torch.no_grad():
            outputs = self.model(img)
            
            # Handle different model output types
            if hasattr(self.model, 'config'):  # Transformers model
                if hasattr(outputs, 'logits'):
                    _, pred = torch.max(outputs.logits, 1)
                else:
                    raise ValueError("Unable to extract predictions")
            else:
                _, pred = torch.max(outputs, 1)
            
            return pred.item()

    def attack(self, 
               original_image, 
               target_class, 
               confidence=0, 
               learning_rate=0.01, 
               max_iterations=10, 
               binary_search_steps=9, 
               initial_const=1e-3,
               norm='l2'):
        
        
        original_image = self.preprocess_image(original_image)
        
        original_image = original_image.to(self.device)
        batch_size = original_image.size(0)
        
        modifier = torch.zeros_like(original_image, requires_grad=True)
        
        # Binary search parameters
        const = torch.full((batch_size,), initial_const, device=self.device)
        
        # Best attack results
        best_adversarial = None
        best_loss = float('inf')
        # print("hello1")
        
        for binary_search_step in range(binary_search_steps):
            # Create optimizer for the modifier
            optimizer = optim.Adam([modifier], lr=learning_rate)
            # print("hello2")
            for iteration in range(max_iterations):
                # Perturb the image
                perturbed_image = torch.clamp(original_image + modifier, 0, 1)
                
                # Get model predictions
                outputs = self.model(perturbed_image)
                # print("lplp")
                
                # Compute attack loss
                loss = self._compute_loss(
                    outputs, 
                    target_class, 
                    original_image, 
                    perturbed_image, 
                    const, 
                    confidence, 
                    norm
                )
                # print("opop")
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update best adversarial if needed
                if loss < best_loss:
                    best_loss = loss.item()
                    best_adversarial = perturbed_image.clone().detach()
        
        return best_adversarial
    
   
    def _compute_loss(self, 
                 outputs, 
                 target_class=None, 
                 original_image=None, 
                 perturbed_image=None, 
                 const=1.0, 
                 confidence=0.0, 
                 norm='l2'):
        """
        Compute a custom loss function for the Classifier model
        
        Args:
            outputs (torch.Tensor): Model predictions
            target_class (int, optional): Desired target class for adversarial loss
            original_image (torch.Tensor, optional): Original input image
            perturbed_image (torch.Tensor, optional): Perturbed input image
            const (float): Regularization constant for perturbation
            confidence (float): Margin for classification loss
            norm (str): Norm type for perturbation loss ('l2' or 'linf')
        
        Returns:
            torch.Tensor: Computed loss
        """
        # Ensure outputs are in the correct shape for binary classification
        outputs = outputs.squeeze()
        
        # Default to binary classification with threshold 0.5
        if target_class is None:
            target_class = 1 if outputs > 0.5 else 0
        
        # Classification loss (binary cross-entropy)
        classification_loss = F.binary_cross_entropy_with_logits(
            outputs, 
            torch.tensor(target_class, dtype=torch.float32).to(outputs.device)
        )
        
        # Perturbation loss (if original and perturbed images are provided)
        if original_image is not None and perturbed_image is not None:
            # Compute perturbation norm
            if norm == 'l2':
                perturbation_loss = torch.norm(perturbed_image - original_image)
            elif norm == 'linf':
                perturbation_loss = torch.max(torch.abs(perturbed_image - original_image))
            else:
                raise ValueError(f"Unsupported norm type: {norm}")
            
            # Combined loss with regularization
            total_loss = classification_loss + const * perturbation_loss
        else:
            total_loss = classification_loss
        
        return total_loss

  
    def attack_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]], 
        target_classes: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Perform batch Carlini-Wagner Attack with performance tracking
        """
        # Validate inputs
        if target_classes is not None and len(target_classes) != len(images):
            raise ValueError("Number of target classes must match number of images")

        # Perform attacks
        perturbed_images = []
        start_batch_time = time.time()
        
        for idx, image in enumerate(images):
            iteration_start = time.time()
            
            target_class = target_classes[idx] if target_classes else None
            
            try:
                perturbed_img = self.attack(image, target_class)
                perturbed_images.append(perturbed_img)
                
                # Log individual image attack time
                iteration_time = time.time() - iteration_start
                self.logger.info(f"Image {idx} attack time: {iteration_time:.4f} seconds")
            except Exception as e:
                self.logger.error(f"Attack failed for image {idx}: {e}")
                perturbed_images.append(self.preprocess_image(image))  # Fallback to original image
        
        batch_total_time = time.time() - start_batch_time
        self.logger.info(f"Batch attack total time: {batch_total_time:.4f} seconds")
        
        return perturbed_images


def batch_adversarial_attack_CandW(
    model, 
    dataset_path: str, 
    output_path: str = 'perturbed_images', 
    batch_size: int = 16,
    total_images: int = 1000,
    randomize: bool = True
):
   
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'REAL'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'FAKE'), exist_ok=True)


    attacker = CarliniWagnerAttack(model)

  
    total_generated = 0
    overall_start_time = time.time()
    fooled_count = 0


    for category in ['REAL', 'FAKE']:
        # Early stopping if total images are generated
        if total_generated >= total_images:
            break

        category_path = os.path.join(dataset_path, category)
        image_files = [
            f for f in os.listdir(category_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        # Randomize if specified
        if randomize:
            np.random.shuffle(image_files)

        # Limit images per category proportionally
        category_limit = min(
            len(image_files), 
            max(1, int(total_images * 0.5))  # 50-50 split between REAL and FAKE
        )
        image_files = image_files[:category_limit]

        # Process in batches
        for i in range(0, len(image_files), batch_size):
            # Early stopping check
            if total_generated >= total_images:
                break

            # Determine batch size, ensuring we don't exceed total_images
            current_batch_size = min(
                batch_size, 
                len(image_files) - i,
                total_images - total_generated
            )

            # Slice current batch files
            batch_files = image_files[i:i+current_batch_size]
            
            # Load images for current batch
            images = [
                Image.open(os.path.join(category_path, img_file)).convert('RGB')
                for img_file in batch_files
            ]

            # Perform attacks on current batch
            perturbed_images = attacker.attack_batch(images)

            # Save and check model predictions
            for img_file, orig_img, perturbed_img in zip(batch_files, images, perturbed_images):
                # Save original image
                orig_save_path = os.path.join(output_path, category, f'original_{img_file}')
                orig_img.save(orig_save_path)

                # Save perturbed image 
                perturbed_save_path = os.path.join(output_path, category, f'perturbed_{img_file}')
                save_perturbed_image(perturbed_img, perturbed_save_path)

                # Check original and perturbed image predictions
                orig_pred = get_inference(orig_save_path)
                perturbed_pred = get_inference(perturbed_save_path)

                # Fooling detection (binary classification)
                orig_label = 1 if orig_pred > 0.5 else 0
                perturbed_label = 1 if perturbed_pred > 0.5 else 0

                # Check if model was fooled
                was_fooled = orig_label != perturbed_label
                if was_fooled:
                    fooled_count += 1

                # Log fooling result
                print(f"Image: {img_file}")
                print(f"  Original Prediction: {orig_pred.item():.4f} (Label: {orig_label})")
                print(f"  Perturbed Prediction: {perturbed_pred.item():.4f} (Label: {perturbed_label})")
                print(f"  Fooled: {was_fooled}\n")

                # Update total generated
                total_generated += 1

                # Final stopping condition
                if total_generated >= total_images:
                    break

    overall_time = time.time() - overall_start_time
    print(f"\nAdversarial attack complete.")
    print(f"Total images generated: {total_generated}")
    print(f"Images fooled: {fooled_count}")
    print(f"Fooling rate: {(fooled_count/total_generated)*100:.2f}%")
    print(f"Overall attack time: {overall_time:.4f} seconds")

    return total_generated, fooled_count




# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('hopskip_batch_attack.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def save_perturbed_image(tensor_image, filename='perturbed_image.png'):
   
    reverse_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    img = tensor_image.squeeze(0)
    denormalized = reverse_normalize(img)
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(denormalized.cpu().clamp(0, 1))
    pil_img=pil_img.resize((32,32))
    pil_img.save(filename)

class BatchHopSkipJumpAttack:
    def __init__(
        self, 
        model, 
        device: Optional[torch.device] = None, 
        max_iterations: int = 30, 
        step_size: float = 1.0
    ):
       
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.is_transformers_model = hasattr(model, 'config')

        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),            
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            
        ])

    def preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        
        if isinstance(image, (np.ndarray, Image.Image)):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.preprocess(image).unsqueeze(0)
        
        return image.to(self.device)

    def _get_prediction(self, img):
        
        with torch.no_grad():
            outputs = self.model(img.unsqueeze(0))
            
            # Handle different model output types
            if self.is_transformers_model:
                if hasattr(outputs, 'logits'):
                    _, pred = torch.max(outputs.logits, 1)
                else:
                    raise ValueError("Unable to extract predictions")
            else:
                _, pred = torch.max(outputs, 1)
            
            return pred.item()

    def _calculate_confidence(
        self, 
        model_output, 
        original_pred: int, 
        target_class: Optional[int] = None
    ) -> float:

        if self.is_transformers_model:
            logits = model_output.logits if hasattr(model_output, 'logits') else model_output
        else:
            logits = model_output

        probs = torch.softmax(logits, dim=1)
        
        if target_class is not None:
            return probs[0, target_class].item()
        else:
            return 1 - probs[0, original_pred].item()

    def attack_batch(
    self, 
    images: torch.Tensor, 
    target_classes: Optional[List[int]] = None
        ) -> List[torch.Tensor]:

        original_preds = [1 if pred > 0.5 else 0 for pred in self.model(images).squeeze()]
        
        perturbed_images = []
        for idx, (image, orig_pred) in enumerate(zip(images, original_preds)):
            target_class = target_classes[idx] if target_classes else None
            
            try:
                perturbed_img = self.attack_single_image(image, orig_pred, target_class)
                perturbed_images.append(perturbed_img)
                logger.info(f"Successfully attacked image {idx}")
            except Exception as e:
                logger.error(f"Attack failed for image {idx}: {e}")
                perturbed_images.append(image)  # Fallback to original image
        
        return perturbed_images

    def attack_single_image(
    self, 
    image: torch.Tensor, 
    original_pred: int, 
    target_class: Optional[int] = None
        ) -> torch.Tensor:
       
        device = next(self.model.parameters()).device
        image = image.to(device)
        
        # Prepare the image for attack
        image_original = image.clone().detach()
        best_perturbation = image.clone()
        best_confidence = float('-inf')
        
        
        epsilon = 0.1  
        alpha = 0.02 
        
        for _ in range(self.max_iterations):
           
            image.requires_grad = True
            if image.grad is not None:
                image.grad.zero_()
            
            outputs = self.model(image.unsqueeze(0))
            
            # Calculate loss
            if target_class is not None:
                loss = -F.binary_cross_entropy(outputs, torch.tensor([[1.0] if target_class == 1 else [0.0]]).to(device))
            else:
                loss = F.binary_cross_entropy(outputs, torch.tensor([[0.0] if original_pred == 0 else [1.0]]).to(device))
            loss.backward()
            
            with torch.no_grad():
                perturbed_image = image + alpha * torch.sign(image.grad)
                perturbed_image = torch.clamp(
                    perturbed_image, 
                    min=torch.clamp(image_original - epsilon, 0, 1),
                    max=torch.clamp(image_original + epsilon, 0, 1)
                )
            
            # Evaluate new perturbation
            with torch.no_grad():
                new_outputs = self.model(perturbed_image.unsqueeze(0))
                
                # Confidence calculation for binary classification
                confidence = torch.abs(new_outputs - 0.5)
                
                # Update best perturbation if it increases misclassification confidence
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_perturbation = perturbed_image.clone()
            
            # Update image for next iteration
            image = perturbed_image.clone().detach()
        
        return best_perturbation.detach()



def test_HOPSKI_attack_batch(model, test_images, test_labels, batch_size=16, iterations=40, step_size=1, device='cuda'):
    attacker = BatchHopSkipJumpAttack(model, max_iterations=iterations, step_size=step_size)

   
    if isinstance(test_images, list):
        test_images = torch.stack(test_images)
    
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    global img_id, failed_num
    failed_num = 0

    # Generate adversarial examples
    adversarial_images = attacker.attack_batch(test_images)

    # Evaluate predictions
    original_logits = model(test_images).squeeze()
    adversarial_logits = model(torch.stack(adversarial_images)).squeeze()

    original_preds = (original_logits > 0.5).float()
    adversarial_preds = (adversarial_logits > 0.5).float()

    for j in range(len(test_images)):
        # Visualization and saving logic remains the same
        adversarial_image_np = denormalize_image_vit(adversarial_images[j])
        adversarial_image_np1 = Image.fromarray(adversarial_image_np)
        adversarial_image_np1_resized = adversarial_image_np1.resize((32, 32))

        image_real_np = denormalize_image_vit(test_images[j])
        real_image_np1 = Image.fromarray(image_real_np)
        real_image_np1_resized = real_image_np1.resize((32, 32))

        label_str = "REAL" if test_labels[j] == 0 else "FAKE"
        
        adversarial_image_np1_resized.save(f'/workspace/adversarial-attack-generation/adversarial_test_images_HopSkiattack/{label_str}/{img_id}.png')
        real_image_np1_resized.save(f'/workspace/adversarial-attack-generation/adversarial_test_images_HopSkiattack/{label_str}/{img_id}_original.png')
        img_id += 1

        # Check if the adversarial image fooled the model
        if original_preds[j] != adversarial_preds[j]:
            failed_num += 1
            print(f"Success: Image {j} fooled the model")
        else:
            print(f"Failure: Image {j} did not fool the model")
    fooling_rate = failed_num / len(test_images)
    print(f"Fooling Rate: {fooling_rate * 100:.2f}%")


def measure_performance(func):
    
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()
        
        # Start timing
        start_time = time.time()
        start_cpu = time.process_time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # End timing
        end_time = time.time()
        end_cpu = time.process_time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Print performance metrics
        print("\n--- Performance Metrics ---")
        print(f"Wall Clock Time: {end_time - start_time:.4f} seconds")
        print(f"CPU Processing Time: {end_cpu - start_cpu:.4f} seconds")
        print(f"Memory Usage:")
        print(f"  Current Memory: {current / 1024:.2f} MB")
        print(f"  Peak Memory: {peak / 1024:.2f} MB")
        print(f"  Actual Memory Change: {end_memory - start_memory:.2f} MB")
        
        return result
    return wrapper

class DeepFoolAttack:

    def __init__(
        self, 
        model, 
        device: Optional[torch.device] = None, 
        max_iterations: int = 50, 
        overshoot: float = 0.02
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.max_iterations = max_iterations
        self.overshoot = overshoot
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_prediction(self, img):
        with torch.no_grad():
            img = img.to(self.device)
            outputs = self.model(img)
            pred = (outputs > 0.5).float()
            return pred.item()

    def _compute_perturbation(
    self, 
    image: torch.Tensor, 
    original_pred: int, 
    target_class: int
) -> torch.Tensor:
        image = image.clone().detach().requires_grad_(True)
        outputs = self.model(image)
        assert outputs.shape == (1, 1), f"Unexpected outputs shape: {outputs.shape}"
        loss_fn = nn.BCELoss()
        target = torch.tensor([[float(target_class)]], device=self.device, dtype=torch.float32)
        assert target.shape == (1, 1), f"Unexpected target shape: {target.shape}"
        loss = loss_fn(outputs, target)
        loss.backward()
        grad = image.grad.data.clone()
        if grad is None:
            print("Gradient is None. Skipping perturbation.")
            return torch.zeros_like(image)
        return grad
    def attack(
    self, 
    image: Union[Image.Image, np.ndarray, torch.Tensor], 
    target_class: Optional[int] = None
) -> torch.Tensor:
        original_image = self.preprocess_image(image)
        if original_image is None:
            return None
        original_pred = self._get_prediction(original_image)
        if target_class is None:
            target_class = 1 - original_pred
        else:
            target_class = int(target_class)
        perturbed_image = original_image.clone().detach().requires_grad_(True)
        for _ in range(self.max_iterations):
            outputs = self.model(perturbed_image)
            current_pred = self._get_prediction(perturbed_image)
            if current_pred == target_class:
                break
            perturbation = self._compute_perturbation(perturbed_image, original_pred, target_class)
            perturbed_image = perturbed_image + (1 + self.overshoot) * perturbation
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image.detach()
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
       
        try:
        
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if isinstance(image, Image.Image):
                image = self.preprocess(image)
            
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
          
            return image.to(self.device)
        
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            raise

    @measure_performance
    def attack_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]], 
        target_classes: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        
     
        if target_classes and len(target_classes) != len(images):
            target_classes = None  # Fall back to automatic targeting
        
        # Perform attacks
        perturbed_images = []
        for idx, image in enumerate(images):
            try:
               
                target_class = target_classes[idx] if target_classes else None
               
                perturbed_img = self.attack(image, target_class)
                perturbed_images.append(perturbed_img)
            
            except Exception as e:
                self.logger.error(f"Batch attack failed for image {idx}: {e}")
                # Fallback: use original image
                perturbed_images.append(self.preprocess_image(image))
        
        return perturbed_images



@measure_performance
def batch_adversarial_attack_Deepfool(
    model, 
    dataset_path: str, 
    output_path: str = 'perturbed_images', 
    batch_size: int = 16,
    total_images: int = 1000,
    randomize: bool = True
):
  
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'REAL'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'FAKE'), exist_ok=True)


    attacker = DeepFoolAttack(model=model)

    # Track total images generated
    total_generated = 0

    # Process each category
    for category in ['REAL', 'FAKE']:
        if total_generated >= total_images:
            break

        category_path = os.path.join(dataset_path, category)
        image_files = [
            f for f in os.listdir(category_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        if randomize:
            np.random.shuffle(image_files)


        category_limit = min(
            len(image_files), 
            max(1, int(total_images * 0.5))  # 50-50 split
        )
        image_files = image_files[:category_limit]
        for i in range(0, len(image_files), batch_size):
      
            if total_generated >= total_images:
                break


            current_batch_size = min(
                batch_size, 
                len(image_files) - i,
                total_images - total_generated
            )
            batch_files = image_files[i:i+current_batch_size]
            
            images = [
                Image.open(os.path.join(category_path, img_file)).convert('RGB')
                for img_file in batch_files
            ]


            print(f"\nProcessing {category} - Batch {i//batch_size + 1}")
            print(f"Current batch size: {len(images)}")
            print(f"Total images generated so far: {total_generated}")


            try:
                perturbed_images = attacker.attack_batch(images)

                for img_file, perturbed_img in zip(batch_files, perturbed_images):
                    save_path = os.path.join(output_path, category, f'perturbed_{img_file}')
                    save_perturbed_image(perturbed_img, save_path)
                    
                    total_generated += 1
                    if total_generated >= total_images:
                        print(f"\nReached target of {total_images} images. Stopping.")
                        break

            except Exception as e:
                print(f"Batch attack failed: {e}")
                continue

    print(f"\nAdversarial attack complete.")
    print(f"Total images generated: {total_generated}")

    return total_generated
def preprocess_imageDeepfool(image_path):
        try:
            image = Image.open(image_path)
            if hasattr(image, 'n_frames') and image.n_frames > 1:
                print(f"Image {image_path} has multiple frames. Skipping.")
                return None
            image = image.convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None
        tensor = preprocess(image).unsqueeze(0).to(device)
        return tensor

def denormalizeDeepfool(tensor, mean=MEAN, std=STD):
    tensor = tensor.clone()
    tensor = tensor * torch.tensor(std, device=tensor.device).view(3, 1, 1) + torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    return tensor

def save_imageDeepfool(tensor, path, mean=MEAN, std=STD):
    assert tensor.dim() == 4 and tensor.shape[0] == 1, "Tensor must be a 4D tensor with batch size 1"
    tensor = denormalizeDeepfool(tensor, mean, std)
    tensor = tensor.cpu()
    tensor = tensor.squeeze(0)
    img_pil = transforms.ToPILImage()(tensor)
    img_pil.save(path)
def test_adversarial_imagesDeepfool(model, attacker, image_paths, labels, device):
    results = []
    fooled_count = 0
    for img_path, label in zip(image_paths, labels):
        try:
            original_image = preprocess_imageDeepfool(img_path)
            if original_image is None:
                continue
            perturbed_image = attacker.attack(original_image, target_class=1 - label)
            if perturbed_image is None or perturbed_image.dim() != 4:
                continue
            with torch.no_grad():
                original_output = model(original_image)
                perturbed_output = model(perturbed_image)

            original_pred = (original_output > 0.5).int().item()
            perturbed_pred = (perturbed_output > 0.5).int().item()

            fooled = perturbed_pred != original_pred
            if fooled:
                fooled_count += 1
            save_dir = os.path.join(STORING_DIR, 'REAL' if label == 0 else 'FAKE')
            save_name = f"{'fooled_' if fooled else 'not_fooled_'}{os.path.basename(img_path)}"
            save_path = os.path.join(save_dir, save_name)
            save_imageDeepfool(perturbed_image, save_path)
            results.append((img_path, original_pred, perturbed_pred, fooled))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    return results, fooled_count
def RunDeepfool(length=16):
    REAL_DIR = f'{DATASET_PATH}/REAL'
    FAKE_DIR = f'{DATASET_PATH}/FAKE'


    # Number of images to select from each class
    NUM_IMAGES_PER_CLASS =int( length/2)
    TOTAL_IMAGES = NUM_IMAGES_PER_CLASS * 2    

    os.makedirs(os.path.join(STORING_DIR, 'REAL'), exist_ok=True)
    os.makedirs(os.path.join(STORING_DIR, 'FAKE'), exist_ok=True)


    real_images = [os.path.join(REAL_DIR, img) for img in os.listdir(REAL_DIR) if img.endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [os.path.join(FAKE_DIR, img) for img in os.listdir(FAKE_DIR) if img.endswith(('.png', '.jpg', '.jpeg'))]

    if len(real_images)>0:
        selected_real = random.sample(real_images, NUM_IMAGES_PER_CLASS)
    else:
        selected_real=[]
    if len(fake_images)>0:
        selected_fake = random.sample(fake_images, NUM_IMAGES_PER_CLASS)
    else:
        selected_fake=[]

       

    image_paths = selected_real + selected_fake
    labels = [0] * NUM_IMAGES_PER_CLASS + [1] * NUM_IMAGES_PER_CLASS  # 0 for REAL, 1 for FAKE
    

    attacker=DeepFoolAttack(model)
    results, fooled_count = test_adversarial_imagesDeepfool(model, attacker, image_paths, labels, device)

    # Display results
    for img_path, original_pred, perturbed_pred, fooled in results:
        print(f"Image: {img_path}")
        print(f"Original Prediction: {'FAKE' if original_pred else 'REAL'}")
        print(f"Perturbed Prediction: {'FAKE' if perturbed_pred else 'REAL'}")
        print(f"Fooling Successful: {'Yes' if fooled else 'No'}")
        print("-" * 30)

    # Calculate fooling rate
    total_processed = len(results)
    if total_processed > 0:
        fooling_rate = (fooled_count / total_processed) * 100
    else:
        fooling_rate = 0.0
    print(f"Total Images Processed: {total_processed}")
    print(f"Number of Fooling Images: {fooled_count}")
    print(f"Fooling Rate: {fooling_rate:.2f}%")    



if ATTACK_NAME=="PGD":
    batch_size_num=BATCH_SIZE # set batch size accordingly
    label_map = {"REAL": 0, "FAKE": 1}
    test_images,test_labels=load_images_from_path("/workspace/archive/test",label_map,batch_size=batch_size_num)
    test_pgd_attack_batch(model, test_images[0][:LENGTH], test_labels[0][:LENGTH], batch_size=batch_size_num, epsilon=0.03, alpha=0.02, iterations=30)
    print("Number of images that were failed to fool are " ,failed_num)
elif ATTACK_NAME=="Spatial":
    generate_adversarial_images(
    dataset_path=DATASET_PATH,
    model=model,  # Your trained model
    num_images=LENGTH,
    batch_size=BATCH_SIZE,
    max_rotation=10,# set this accordingly 
    max_translation=0.05,# set this accordingly
    output_dir='adversarial_test_images_Spatial',# output dir
    recursive=True  # Search for images in subdirectories
    )
elif ATTACK_NAME=="C&W":
    batch_adversarial_attack_CandW(
        model, 
        DATASET_PATH, 
        STORING_DIR, # set the output path dir 
        batch_size=BATCH_SIZE,     # Process BATCH_SIZE images per batch
        total_images=LENGTH,  # Generate LENGTH total perturbed images
        randomize=True     # Randomize image selection
    )
elif ATTACK_NAME=="HopSkipJump":
    batch_size_num=BATCH_SIZE
    label_map = {"REAL": 0, "FAKE": 1}
    test_images,test_labels=load_images_from_path("/workspace/archive/test",label_map,batch_size=batch_size_num)
    test_HOPSKI_attack_batch(model, test_images[0][:LENGTH], test_labels[0][:LENGTH], batch_size=batch_size_num, iterations=20,step_size=2)
elif ATTACK_NAME=="DeepFool":
    RunDeepfool(LENGTH)
else:
    print("Invalid attack name ")    





