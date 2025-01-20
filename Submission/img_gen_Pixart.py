"""
This python file generate Images using Pixart Sigma
"""

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, PixArtSigmaPipeline
import time
import math
import torch
import os
import random

#----------------------------------------------#
#GPU IS REQUIRED FOR COMPUTING           #
#install torch 2.2.0 with stable fast          #
#----------------------------------------------#

#Constants
NUM_IMAGES_PER_CLASS = 500
BSZ = 4
startind = 500
OUTPUT_DIR =""  # Provide Path to Output Directory


def load_model():
    """
    Load the PixArt Sigma Pipeline model.

    Returns:
        The loaded model.
    """

    model = PixArtSigmaPipeline.from_pretrained(
        'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        torch_dtype=torch.float16,
        cache_dir="."
    )
    model.to(torch.device('cuda'))
    return model

def generate_prompts(class_number, batch_size=1):
    """
    Generate prompts for a specific class with modifiers.

    Args:
        class_number (int): Index of the class for which to generate prompts.
        batch_size (int): Number of prompts to generate.

    Returns:
        list: List of generated prompts.
    """
    classes = [
        ('Airplane', ['aircraft', 'airplane', 'fighter', 'flying', 'jet', 'plane']),
        ('Automobile', ['sedan', 'new', 'sports', 'vintage',"SUV"]),
        ('Bird', ['flying', 'in a tree', 'indoors', 'on water', 'outdoors', 'walking']),
        ('Cat', ['indoors', 'outdoors', 'walking', 'running', 'eating', 'jumping', 'sleeping', 'sitting']),
        ('Deer', ['herd', 'in a field', 'in the forest', 'outdoors', 'running', 'wildlife photography']),
        ('Dog', ['indoors', 'outdoors', 'walking', 'running', 'eating', 'jumping', 'sleeping', 'sitting']),
        ('Frog', ['European', 'in the forest', 'on a tree', 'on the ground', 'swimming', 'tropical', 'wildlife photography']),
        ('Horse', ['herd', 'in a field', 'in the forest', 'outdoors', 'running', 'wildlife photography']),
        ('Ship', ['at sea', 'boat', 'cargo', 'cruise', 'on the water', 'river', 'sailboat', 'tug']),
        ('Truck', ['18-wheeler', 'car transport', 'firetruck'])
    ]
    
    if class_number < 0 or class_number >= len(classes):
        raise ValueError(f"Invalid class number. Must be between 0 and {len(classes)-1}")
    
    class_name, modifiers = classes[class_number]
    
    article = 'an' if class_name[0].lower() in 'aeiou' else 'a'
    
    prompts = []
    for _ in range(batch_size):
        
        selected_modifier = random.choice(modifiers)
        
        
        prompt = f"a high quality photograph of {article} {class_name.lower()} {selected_modifier}"
        prompts.append(prompt)
    
    return prompts

#----------------------------------------------------compiling-------------------------------------------------------

model = load_model()
model.transformer.forward = torch.compile(model.transformer.forward,mode="max-autotune")
model.text_encoder.forward = torch.compile(model.text_encoder.forward,mode="max-autotune")
model.vae.forward = torch.compile(model.vae.forward,mode="max-autotune")
kwarg_inputs = dict(
    prompt='(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
    height=1024,
    width=1024,
    num_inference_steps=30,
    num_images_per_prompt=4,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
for _ in range(1):
    output_image = model(**kwarg_inputs).images[0]




#-------------------------------------------------Image Generation----------------------------------------

for classname in range(2,10):
    num_batches = math.ceil(NUM_IMAGES_PER_CLASS/BSZ)
    os.makedirs(f"{OUTPUT_DIR}/{classname}",exist_ok=True)
    for i in range(0,num_batches):
        prompts = generate_prompts(classname,BSZ)

        images = model(
            prompt=prompts,
            negative_prompt = ["poor quality, blurry, cropped, deformed"]*BSZ,
            height = 1024,
            width = 1024,
            num_inference_steps=30,
            num_images_per_prompt=1
        ).images

        for j in range(BSZ):
            images[j].save(f"{OUTPUT_DIR}/{classname}/{startind+i*BSZ+j}.png")


    




