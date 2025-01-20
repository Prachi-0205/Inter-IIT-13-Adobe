#######Requirements##########
#pip install unsloth#
# CUDA_VISIBLE_DEVICES=2 python3 pixtral.py
# CUDA_VISIBLE_DEVICES=3 python3 qwen.py


#----------------------------------------------#
#EXTRA GPU IS REQUIRED FOR COMPUTING           #
#install torch 2.5.1 with Unsloth as dependency#
#----------------------------------------------#
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
from unsloth import FastVisionModel 
import torch
import wandb
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig



#use only the below provided key
HF_LOGIN_KEY = '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'
HF_DATASET = "22-24/Final"
WANDB_API_KEY = ""   #Provide the WANDB API KEY
login(token=HF_LOGIN_KEY)





def resize_image(example):
    """
    Resizes the input image to 32x32, then to 512x512.
    
    Parameters:
        example (dict): Contains 'image' as a PIL.Image object or file path.
    
    Returns:
        dict: Updated dictionary with resized image.
    """
    
    if isinstance(example['image'], Image.Image):
        image = example['image']
    else:
        image = Image.open(example['image'])  

    
    image = image.resize((32, 32))
    image = image.resize((512,512))
    example['image'] = image
    return example

dataset = load_dataset(HF_DATASET, split = "train")
dataset = dataset.map(resize_image, batched=False)

model, tokenizer = FastVisionModel.from_pretrained(
   
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",
    load_in_4bit = False, 
    use_gradient_checkpointing = "unsloth", 

)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 

    r = 16,          
    lora_alpha = 16, 
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
    
)


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
def convert_to_conversation(sample):
    """
    Converts a sample to a conversation format.

    Parameters:
        sample (dict): Dictionary containing 'image' and 'answer'.

    Returns:
        dict: Formatted conversation with user and assistant messages.
    """

    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : system_message},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["answer"]} ]
        },
    ]
    return { "messages" : conversation }
pass


converted_dataset = [convert_to_conversation(sample) for sample in dataset]




wandb.login(key=WANDB_API_KEY)


#--------------------------------------------------------------Training--------------------------------------------------------#


FastVisionModel.for_training(model) 

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        # max_steps = 30,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 3e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.065,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "Pixtral",
        report_to = "wandb",     # For Weights and Biases

        
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)


trainer_stats = trainer.train()
model.push_to_hub("22-24/pixtral_2", token = HF_LOGIN_KEY) # Online saving
tokenizer.push_to_hub("22-24/pixtral_2", token = HF_LOGIN_KEY) # Online saving