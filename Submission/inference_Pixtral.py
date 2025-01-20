from huggingface_hub import login
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import FastVisionModel
from PIL import Image
from transformers import TextStreamer

"""
This is the inference script for Pixtral
The final Model was uploaded on hugging face.
- In the `inference.py` use only the `HF_LOGIN_KEY` provided below.
- Provide the Path to image for inference at `IMAGE_PATH`

"""

HF_LOGIN_TOKEN = "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"  #Use This Token ONLY
IMAGE_PATH = ""   #Provide Path to Image

login(HF_LOGIN_TOKEN)






model, tokenizer = FastVisionModel.from_pretrained(
    
    "22-24/pixtral_2", # Chooose between "22-24/pixtral", "22-24/pixtral_2"
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
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
YOU MUST GIVE ATLEAST 5 ARTIFACTS AND AT MOST 15 ARTIFACTS"""




FastVisionModel.for_inference(model) 
image = Image.open(IMAGE_PATH)  
image = image.resize((512,512))




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

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024,
                   use_cache = True, temperature = 0.6,min_p=0.1)