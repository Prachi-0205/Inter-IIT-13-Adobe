"""
This python file generate Images using table Diffusion 1.5  Model
"""

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, PixArtSigmaPipeline
import math
import torch
from diffusers import (StableDiffusionPipeline, EulerAncestralDiscreteScheduler)
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
import os
import random

#Constants
NUM_IMAGES_PER_CLASS = 547
BSZ = 4
startind = 452

OUTPUT_DIR="" # Give path to Output Dir

def load_model():
    """
    Load the Stable Diffusion 1.5  Model

    Returns:
        The loaded model.
    """
    model = StableDiffusionPipeline.from_pretrained(
        'SG161222/Realistic_Vision_V6.0_B1_noVAE',
        torch_dtype=torch.float16,cache_dir=".")

    # model.scheduler = EulerAncestralDiscreteScheduler.from_config(
    #     model.scheduler.config)
    model.safety_checker = None
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
config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
try:
    import xformers
    config.enable_xformers = True
except ImportError:
    print('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
except ImportError:
    print('Triton not installed, skip')

config.enable_cuda_graph = True  #Enable CUDA fro small batches

model = compile(model, config)

kwarg_inputs = dict(
    prompt='(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=30,
    num_images_per_prompt=4,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
for _ in range(1):
    output_image = model(**kwarg_inputs).images[0]

#-------------------------------------------------Image Generation------------------------------------------------

for classname in range(0,10):
    num_batches = math.ceil(NUM_IMAGES_PER_CLASS/BSZ)
    os.makedirs(f"{OUTPUT_DIR}/{classname}",exist_ok=True)  
    for i in range(0,num_batches):
        prompts = generate_prompts(classname,BSZ)

        images = model(
            prompt=prompts,
            negative_prompt = ["poor quality, blurry, cropped, deformed"]*BSZ,
            height = 512,
            width = 512,
            num_inference_steps=30,
            guidance_scale=4,
            num_images_per_prompt=1
        ).images

        for j in range(BSZ):
            images[j].save(f"{OUTPUT_DIR}/{classname}/{startind+i*BSZ+j}.png")


    




