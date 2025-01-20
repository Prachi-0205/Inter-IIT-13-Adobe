import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import pywt
from PIL import Image
import os
from PIL import Image
from torchvision import transforms
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from unsloth import FastVisionModel



"""
This is the final pipeline for our Model 
"""

FFT = True   # do not touch if model is not FFT based
IN_CHANS=3 if not FFT else 9
RESIZE_HEIGHT = 224 #input size for model
RESIZE_WIDTH = 224
NUM_SAVE_STEPS=100
MODEL = "coatnet_nano_rw_224"  #Model Name
CKPT_PATH= f"coatnet.pt"  #path to the checkpoints model for TASK 1
SECOND_MODEL_LOADED =0
DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-07-23"
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224

data_transforms = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


HF_LOGIN_TOKEN = "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"
IMAGE_PATH = "/workspace/archive/train/FAKE/1000 (2).jpg"   #Provide Path to Image

login(HF_LOGIN_TOKEN)



class FeatureExtractor(nn.Module):
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
        # Compute FFT
        fft_complex = torch.fft.fft2(x)
        fft_complex_shifted = torch.fft.fftshift(fft_complex)

        # Apply filters
        low_pass = self.apply_low_pass_filter(fft_complex_shifted)
        high_pass = self.apply_high_pass_filter(fft_complex_shifted)

        # Inverse FFT to get filtered images
        low_pass_ifft = torch.fft.ifft2(torch.fft.ifftshift(low_pass)).abs()
        high_pass_ifft = torch.fft.ifft2(torch.fft.ifftshift(high_pass)).abs()

        # Extract magnitude and phase
        low_pass_magnitude = torch.abs(low_pass_ifft)
        high_pass_magnitude = torch.abs(high_pass_ifft)

        # Normalize features
        low_pass_magnitude = low_pass_magnitude / torch.max(low_pass_magnitude)
        high_pass_magnitude = high_pass_magnitude / torch.max(high_pass_magnitude)

        # Concatenate low-pass and high-pass
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
        fft_features = self.extract_fft_features(x)       # Shape: (B, 6, 32, 32)
        # wavelet_features = self.extract_wavelet_features(x)  # Shape: (B, 12, 32, 32)
        combined_features = torch.cat((x, fft_features.to(x.device,x.dtype)), dim=1)
        return combined_features
    
class Classifier(nn.Module):
    def __init__(self,model,subnets=False):
        super().__init__()
        self.bb = timm.create_model(model,pretrained=False,in_chans=IN_CHANS,features_only=True)
        self.head = nn.Sequential(
                nn.Conv2d(512,128,3,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*9,32),
                nn.ReLU(),
                nn.Linear(32,1),
            )
        
            
    def forward(self, x, temperature=1):
        features = self.bb(x)
        out = F.sigmoid(self.head(features[-1])/temperature)
        

        return out
        


def predict_image(image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predicts the output of the given image using the specified model.

    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the input image.
        device (str): Device to perform computations ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The output of the model for the given image.
    """
    # Define the image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # Normalize
    ])
    
    # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    # input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move model and tensor to the specified device
    # model = model.to(device)
    # input_tensor = input_tensor.to(device)

    # Set the model to evaluation mode
    # model.eval()

    
    model = Classifier(MODEL).to("cuda")
    model.eval()
    if image_path.size[0]>32:
        image_path = image_path.resize((32,32))
    img = data_transforms(image_path).unsqueeze(0).to(device)
    fft_extractor = FeatureExtractor(4,224)
    if FFT:
        imgs = fft_extractor(img.to("cuda"))
    else:
        imgs = img.to("cuda")
    
    model.load_state_dict(torch.load(CKPT_PATH,weights_only=True))
        
    with torch.no_grad():  # Disable gradient calculations for inference
        
        output = model(imgs)

        if output.item()>0.5:
            return 1
        else:
            return 0
            
    
def load_second_model():
    model2, tokenizer = FastVisionModel.from_pretrained(
        
        "22-24/pixtral_2",
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    SECOND_MODEL_LOADED = 1
    return model2, tokenizer 

def load_checkpoint(CHECKPOINT_PATH, model, optimizer=None):
     checkpoint = torch.load(CHECKPOINT_PATH)
     model.load_state_dict(checkpoint["model_state_dict"])
     print(f"Checkpoint loaded from {CHECKPOINT_PATH}")

     if optimizer is not None:
         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
         print("Optimizer state loaded")

def create_prompt():
    return {
                    "question": """You are an expert AI Image detector: you analyse,detect and report AI-generated artifacts in images with great accuracy and precision, in a trustworthy manner. For each artifact, you provide a general localisation (eg top-left, bottom-right,) with respect to position in the image. The list of artifacts is given to you below, Analyse the given AI-generated image for artifacts and report their exact locations in the image. The image may have more than one artifact. Return all of them, carefully making sure that the perceived artifacts actually exist, and give reasons for the detection of each artifact. Return your outputs in the form, as follows: 
                    
                    {

                    "Artifact Class Name" : "reason",
                    "Artifact Class Name" : "reason",
                    "Artifact Class Name" : "reason",
                    ... 
                    "Artifact Class Name" : "reason",

                    }

List of artifacts :    
- Inconsistent object boundaries
- Discontinuous surfaces
- Non-manifold geometries in rigid structures
- Floating or disconnected components
- Asymmetric features in naturally symmetric objects 
- Misaligned bilateral elements in animal faces 
- Irregular proportions in mechanical components 
- Texture bleeding between adjacent regions
- Texture repetition patterns
- Over-smoothing of natural textures 
- Artificial noise patterns in uniform surfaces
- Unrealistic specular highlights
- Inconsistent material properties
- Metallic surface artifacts 
- Dental anomalies in mammals 
- Anatomically incorrect paw structures
- Improper fur direction flows
- Unrealistic eye reflections
- Misshapen ears or appendages
- Impossible mechanical connections
- Inconsistent scale of mechanical parts
- Physically impossible structural elements
- Inconsistent shadow directions
- Multiple light source conflicts
- Missing ambient occlusion
- Incorrect reflection mapping
- Incorrect perspective rendering
- Scale inconsistencies within single objects
- Spatial relationship errors
- Depth perception anomalies
- Over-sharpening artifacts
- Aliasing along high-contrast edges
- Blurred boundaries in fine details
- Jagged edges in curved structures
- Random noise patterns in detailed areas
- Loss of fine detail in complex structures
- Artificial enhancement artifacts
- Incorrect wheel geometry
- Implausible aerodynamic structures
- Misaligned body panels
- Impossible mechanical joints
- Distorted window reflections
- Anatomically impossible joint configurations
- Unnatural pose artifacts
- Biological asymmetry errors
- Regular grid-like artifacts in textures
- Repeated element patterns
- Systematic color distribution anomalies
- Frequency domain signatures
- Color coherence breaks
- Unnatural color transitions
- Resolution inconsistencies within regions
- Unnatural Lighting Gradients
- Incorrect Skin Tones
- Fake depth of field
- Abruptly cut off objects
- Glow or light bleed around object boundaries
- Ghosting effects: Semi-transparent duplicates of elements
- Cinematization Effects
- Excessive sharpness in certain image regions
- Artificial smoothness
- Movie-poster like composition of ordinary scenes
- Dramatic lighting that defies natural physics
- Artificial depth of field in object presentation
- Unnaturally glossy surfaces
- Synthetic material appearance
- Multiple inconsistent shadow sources
- Exaggerated characteristic features
- Impossible foreshortening in animal bodies
- Scale inconsistencies within the same object class
                    """
                   
        }


list_of_artifact = """"
- Inconsistent object boundaries
- Discontinuous surfaces
- Non-manifold geometries in rigid structures
- Floating or disconnected components
- Asymmetric features in naturally symmetric objects 
- Misaligned bilateral elements in animal faces 
- Irregular proportions in mechanical components 
- Texture bleeding between adjacent regions
- Texture repetition patterns
- Over-smoothing of natural textures 
- Artificial noise patterns in uniform surfaces
- Unrealistic specular highlights
- Inconsistent material properties
- Metallic surface artifacts 
- Dental anomalies in mammals 
- Anatomically incorrect paw structures
- Improper fur direction flows
- Unrealistic eye reflections
- Misshapen ears or appendages
- Impossible mechanical connections
- Inconsistent scale of mechanical parts
- Physically impossible structural elements
- Inconsistent shadow directions
- Multiple light source conflicts
- Missing ambient occlusion
- Incorrect reflection mapping
- Incorrect perspective rendering
- Scale inconsistencies within single objects
- Spatial relationship errors
- Depth perception anomalies
- Over-sharpening artifacts
- Aliasing along high-contrast edges
- Blurred boundaries in fine details
- Jagged edges in curved structures
- Random noise patterns in detailed areas
- Loss of fine detail in complex structures
- Artificial enhancement artifacts
- Incorrect wheel geometry
- Implausible aerodynamic structures
- Misaligned body panels
- Impossible mechanical joints
- Distorted window reflections
- Anatomically impossible joint configurations
- Unnatural pose artifacts
- Biological asymmetry errors
- Regular grid-like artifacts in textures
- Repeated element patterns
- Systematic color distribution anomalies
- Frequency domain signatures
- Color coherence breaks
- Unnatural color transitions
- Resolution inconsistencies within regions
- Unnatural Lighting Gradients
- Incorrect Skin Tones
- Fake depth of field
- Abruptly cut off objects
- Glow or light bleed around object boundaries
- Ghosting effects: Semi-transparent duplicates of elements
- Cinematization Effects
- Excessive sharpness in certain image regions
- Artificial smoothness
- Movie-poster like composition of ordinary scenes
- Dramatic lighting that defies natural physics
- Artificial depth of field in object presentation
- Unnaturally glossy surfaces
- Synthetic material appearance
- Multiple inconsistent shadow sources
- Exaggerated characteristic features
- Impossible foreshortening in animal bodies
- Scale inconsistencies within the same object class
"""
system_message = f"""
Analyze this AI-generated image in detail and identify any artifacts present. Focus on both the background and the objects, ensuring a thorough examination of all areas. Only report artifacts that you are confident are present in the image. Do not list artifacts that are absent or uncertain.
Background Analysis Instructions:
The background must be treated as an essential part of the image. Carefully inspect the background explicitly for artifacts, which commonly include:
Texture repetition patterns (e.g., repeated grass, sky, or road textures)
Artificial noise (e.g., pixelated or unnatural noise in smooth surfaces)
Texture bleeding (e.g., colors or details merging between adjacent regions unnaturally)
Fake depth of field (e.g., abrupt transitions between sharp and blurred areas)
Blurred boundaries (e.g., unclear transitions between background elements)
Ensure that all detected background artifacts are reported. If the background contains grass, roads, or sky, these artifacts should be checked with special attention.
Focus Instructions Based on Image Content:
If the image contains an animal (e.g., deer, horse, bird, frog), prioritize detection of biological/anatomical artifacts (e.g., joint configurations, fur direction, paws, dental, face assymetry etc.) and texture, color, light, and shadow-related artifacts, along with a thorough inspection of the background.
If the image contains a vehicle or mechanical object (e.g., ship, truck, plane), prioritize detection of mechanical/vehicular artifacts (e.g., structural inconsistencies, impossible mechanical connections etc.) and texture, color, light, and shadow-related artifacts, along with a thorough inspection of the background.
In all cases, the background must be inspected for the artifacts listed above.
MAKE SURE THAT YOU INSPECT EVERY ARTIFACT NAME, AND IF YOU FIND ANY ARTIFACT IN THE IMAGE, RETURN THE MOST SIMILAR AND CORRESPONDING POINT FROM THE LIST
Artifacts to Detect:
{list_of_artifact}
Instructions:
Report only the artifacts that are definitively present in the image.
Exclude artifacts you are unsure of or that are not present.
Ensure texture, color, light, and shadow-related artifacts are evaluated and reported for all images.
Tailor your focus on biological/anatomical artifacts if animals are present and on mechanical/vehicular artifacts if vehicles are present.
Explicitly inspect and report background artifacts as listed above, treating the background as an important and integral part of the analysis.
Output Format:  Return a json like so {{
        "Artifact Class Name" : "reason",
        ...
        "Artifact Class Name" : "reason"
        }}
List only the artifacts from the above categories that are detected with certainty. Do not include any artifacts not present in the image.
THIS IS AN AI GENERATED IMAGE, NOT A REAL PHOTOGRAPH, SO THERE MUST BE SOME ARTIFACTS.
YOU MUST GIVE ATLEAST 5 ARTIFACTS AND AT MOST 15 ARTIFACTS
    """

model2, tokenizer = load_second_model()
FastVisionModel.for_inference(model2) 
def secondary_function(image):
    image = Image.open(IMAGE_PATH)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": system_message}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    result = model2.generate(**inputs,
                            max_new_tokens=1024,
                            use_cache=True,
                            temperature=1.5,
                            min_p=0.1)

    # Decode the result to get the generated text
    generated_text = tokenizer.decode(result[0], skip_special_tokens=True)

    generated_text = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    
    return generated_text.replace("```","").replace("</s>","")


def process_prediction(image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict the output of an image and perform further actions based on the prediction.

    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the input image.
        device (str): Device to perform computations ('cuda' or 'cpu').

    Returns:
        Any: Output from secondary_function or the prediction.
    """
    # Get prediction from the model
    prediction = predict_image(image_path, device)
    
    if prediction == 1:  # Assuming a single scalar output make it 1 for now it is 0
        # Bilaterally resize the image to 512x512
        # image = Image.open(image_path).convert("RGB")
        resized_image = image_path.resize((512, 512), Image.BILINEAR)
        
        # Pass resized image to another function

        print("FAKE", secondary_function(resized_image))
        return "FAKE", secondary_function(resized_image)
    else:
        # Return prediction output directly
        print(prediction, None)
        return "REAL", None




