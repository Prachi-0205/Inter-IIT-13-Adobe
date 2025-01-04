#####Run with venv#####

# Load LoRA weights
from datasets import load_dataset
# from huggingface_hub import login

# login("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")
from unsloth import FastVisionModel
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from transformers import TextStreamer
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth
######Choose the model to be trained######
model, tokenizer = FastVisionModel.from_pretrained(
    #"Xkev/Llama-3.2V-11B-cot",#This is LLAVA-O1
    #"unsloth/Pixtral-12B-Base-2409-bnb-4bit",
    #"unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    #"BarraHome/Mistroll-3.0-CoT-Llama-3.2-11B-Vision-Instruct",
    #"unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    # "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit", # Can change to any version of LLAVA 
    


    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


system_message = """
  Analyse the given AI-generated image for artifacts and report their exact locations in the image. 
  The image may have more than one artifact. Return all of them, carefully making sure that the perceived artifacts 
  actually exist, and give reasons for the detection of each artifact. 
    Return your outputs in the form of a JSON, as follows: 
    {
        "Artifact Class Name" : "reason",
        ...
        "Artifact Class Name" : "reason"
    }

    Categories of artifacts to check for:
    - Ambiguous / color / depth related artifacts:
      - Inconsistent object boundaries
      - Artificial noise patterns in uniform surfaces
      - Unrealistic specular highlights
      - Incorrect perspective rendering
      - Scale inconsistencies within single objects
      - Spatial relationship errors
      - Depth perception anomalies
      - Artificial enhancement artifacts
      - Regular grid-like artifacts in textures
      - Repeated element patterns
      - Systematic color distribution anomalies
      - Frequency domain signatures
      - Color coherence breaks
      - Unnatural color transitions
      - Fake depth of field
      - Abruptly cut off objects
      - Glow or light bleed around object boundaries
      - Ghosting effects: Semi-transparent duplicates of elements
      - Cinematization Effects
      - Artificial smoothness
      - Movie-poster like composition of ordinary scenes
      - Artificial depth of field in object presentation
      - Scale inconsistencies within the same object class
    - Texture related artifacts:
      - Texture bleeding between adjacent regions
      - Texture repetition patterns
      - Over-smoothing of natural textures
      - Metallic surface artifacts
      - Over-sharpening artifacts
      - Aliasing along high-contrast edges
      - Blurred boundaries in fine details
      - Jagged edges in curved structures
      - Loss of fine detail in complex structures
      - Random noise patterns in detailed areas
      - Resolution inconsistencies within regions
      - Synthetic material appearance
      - Excessive sharpness in certain image regions
    - Mechanical artifacts:
      - Physically impossible structural elements
      - Implausible aerodynamic structures
      - Impossible mechanical joints
      - Impossible mechanical connections
      - Inconsistent scale of mechanical parts
      - Floating or disconnected components
      - Asymmetric features in naturally symmetric objects
      - Discontinuous surfaces
      - Non-manifold geometries in rigid structures
      - Irregular proportions in mechanical components
      - Inconsistent material properties
    - Animals related artifacts:
      - Dental anomalies in mammals
      - Anatomically incorrect paw structures
      - Improper fur direction flows
      - Unrealistic eye reflections
      - Misshapen ears or appendages
      - Anatomically impossible joint configurations
      - Unnatural pose artifacts
      - Biological asymmetry errors
      - Impossible foreshortening in animal bodies
      - Misaligned bilateral elements in animal faces
      - Incorrect Skin Tones
    - Light related artifacts:
      - Inconsistent shadow directions
      - Multiple light source conflicts
      - Missing ambient occlusion
      - Incorrect reflection mapping
      - Distorted window reflections
      - Unnatural Lighting Gradients
      - Unnaturally glossy surfaces
      - Dramatic lighting that defies natural physics
      - Multiple inconsistent shadow sources
    - Vehicular artifacts:
      - Incorrect wheel geometry
      - Misaligned body panels

    Analyze the following AI-generated image for artifacts and provide detailed findings.And only give the
    answer from the above artifacts only don't make up any from own you are only allowed to select from the above list only 
    don't come up with anything made up.
    """
def convert_to_conversation(sample):

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

dataset = load_dataset("22-24/Final")

FastVisionModel.for_inference(model) # Enable for inference!

image = dataset["train"][10]["image"].resize((32,32)).resize((512,512))

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
                   use_cache = True, temperature = 1.5, min_p = 0.1)

