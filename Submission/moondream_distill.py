
from torch.utils.data import Dataset
from datasets import load_dataset
import torch.nn.functional as F
from random import shuffle
import io
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display
import os
import torch
from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm



#Constants
EPOCHS = 1
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
LR = 2e-5
USE_WANDB = True  
DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-07-23"
ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729
CHECKPOINT_DIR = "./moondream_checkpoints" #Path to moondream_checkpoints



#-----------------------------------------------------DATASET----------------------------------------------------

list_of_artifacts = """
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
system_prompt = f"""Analyze the above AI-generated image in detail and identify any artifacts present. Focus on both the background and the objects, ensuring a thorough examination of all areas. Only report artifacts that you are confident are present in the image. Do not list artifacts that are absent or uncertain.
Background Analysis Instructions:

1. The background must be treated as an essential part of the image. Carefully inspect the background explicitly for artifacts, which commonly include:
---Texture repetition patterns (e.g., repeated grass, sky, or road textures)
---Artificial noise (e.g., pixelated or unnatural noise in smooth surfaces)
---Texture bleeding (e.g., colors or details merging between adjacent regions unnaturally)
---Fake depth of field (e.g., abrupt transitions between sharp and blurred areas)
---Blurred boundaries (e.g., unclear transitions between background elements)
2. Ensure that all detected background artifacts are reported. If the background contains grass, roads, or sky, these artifacts should be checked with special attention.
3. Focus Instructions Based on Image Content:
---If the image contains an animal (e.g., deer, horse, bird, frog), prioritize detection of biological/anatomical artifacts (e.g., joint configurations, fur direction, paws, dental, face assymetry etc.) and texture, color, light, and shadow-related artifacts, along with a thorough inspection of the background.
---If the image contains a vehicle or mechanical object (e.g., ship, truck, plane), prioritize detection of mechanical/vehicular artifacts (e.g., structural inconsistencies, impossible mechanical connections etc.) and texture, color, light, and shadow-related artifacts, along with a thorough inspection of the background.

In all cases, the background must be inspected for the artifacts listed above.
MAKE SURE THAT YOU INSPECT EVERY ARTIFACT NAME, AND IF YOU FIND ANY ARTIFACT IN THE IMAGE, RETURN THE MOST SIMILAR AND CORRESPONDING POINT FROM THE LIST
Artifacts to Detect:
{list_of_artifacts }

Instructions:
--Report only the artifacts that are definitively present in the image.
--Exclude artifacts you are unsure of or that are not present.
--Ensure texture, color, light, and shadow-related artifacts are evaluated and reported for all images.
--Tailor your focus on biological/anatomical artifacts if animals are present and on mechanical/vehicular artifacts if vehicles are present.
--Explicitly inspect and report background artifacts as listed above, treating the background as an important and integral part of the analysis.

Output Format:
--List only the artifacts from the above categories that are detected with certainty. Do not include any artifacts not present in the image.
--THIS IS AN AI GENERATED IMAGE, NOT A REAL PHOTOGRAPH, SO THERE MUST BE SOME ARTIFACTS.
--YOU MUST GIVE ATLEAST 5 ARTIFACTS AND AT MOST 15 ARTIFACTS
       """


class CaptchaDataset(Dataset):
    """
    A dataset class to handle CAPTCHA images and related QA pairs.
    
    Parameters:
        split (str): The dataset split to use ('train', 'validation', or 'test').
        shuffle_data (bool): Whether to shuffle the dataset during initialization.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Fetches the sample at the given index, processes the image, 
                          and returns the QA data with image details.

    """
    def __init__(self, split='train', shuffle_data=True):
        self.data = load_dataset("22-24/Final", trust_remote_code=True)[split]
        self.data = [self.data[i] for i in range(len(self.data))]
        if shuffle_data:
            shuffle(self.data)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches a sample by index, processes the image (e.g., converts to JPEG if needed),
        and returns the corresponding QA data.
        
        Parameters:
            idx (int): Index of the sample to fetch.
        
        Returns:
            dict: A dictionary with the processed image and QA pairs.
        """
        sample = self.data[idx]
        image = sample["image"]
        if isinstance(image, Image.Image) and image.format == "PNG":      
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG",quality=75)
                buffer.seek(0)
                image = Image.open(buffer)
                image = image.resize((32,32))
                image = image.resize((512,512))
                image.load() 
        return {
            "image": image,  
            "teacher_image": sample["image"],
            "qa": [
                {
                    "question": system_prompt,

                    "answer" : sample["answer"],
                }
            ]
        }

datasets = {
    "train": CaptchaDataset("train", shuffle_data=True),
}


#-------------------------------------------------Functions----------------------------------------------

def collate_fn(batch):
    """
    Collates samples into a batch for DataLoader.
    
    Parameters:
        batch (list): List of samples from the dataset.
    
    Returns:
        tuple: A tuple containing:
            - images (list): List of processed images.
            - tokens (torch.Tensor): Tokenized questions and answers.
            - labels (torch.Tensor): Labels for loss computation (-100 for ignored tokens).
            - attn_mask (torch.Tensor): Attention mask for the model.
    """
    images = [sample['image'] for sample in batch]
    teacher_images = [sample["teacher_image"] for sample in batch]
    images = [moondream.vision_encoder.preprocess(image) for image in images]
    teacher_images = [moondream.vision_encoder.preprocess(teacher_image) for teacher_image in teacher_images]

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for qa in sample['qa']:
            q_t = tokenizer(
                f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = tokenizer(
                f" {qa['answer']}{ANSWER_EOS}",
                add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = max(len(labels) for labels in labels_acc)

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images,
        teacher_images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )

def compute_loss(batch):
    """
    Computes the combined loss for the batch, including vision encoder distillation loss and text model loss.
    
    Parameters:
        batch (tuple): The batch containing:
            - images: Input images for the student model.
            - teacher_images: Corresponding images for the teacher model.
            - tokens: Tokenized text input.
            - labels: Labels for the text model (used for language model loss).
            - attn_mask: Attention mask for text tokens.
    
    Returns:
        tuple: Combined loss and vision distillation loss as a float.
    """
    
    images,teacher_images, tokens, labels, attn_mask = batch

    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    img_embs = moondream.vision_encoder(images)
    
    with torch.no_grad():
        teacher_img_embs = teacher_moondream.vision_encoder(teacher_images)

    vision_loss = F.mse_loss(img_embs,teacher_img_embs)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )


    return outputs.loss + vision_loss,vision_loss.item()

def lr_schedule(step, max_steps):
    """
    Learning rate schedule with warm-up and cosine decay.
    
    Parameters:
        step (int): Current training step.
        max_steps (int): Total number of training steps.
    
    Returns:
        float: Learning rate for the current step.
    """
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

def save_checkpoint(epoch, step, model, optimizer, loss,part):
    """
    Saves the training checkpoint for a specific model component.
    
    Parameters:
        epoch (int): Current epoch number.
        step (int): Current training step.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        loss (float): The loss value at the time of saving.
        part (str): A label indicating the part of the model ('t' for text, 'v' for vision).
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_step_{step}_{part}.pt")
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

dataloaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
}

#-------------------------------------------- model and optimizer------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)


teacher_moondream= AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

sample = datasets['train'][0]
display(sample['image'])

for qa in sample['qa']:
    print('Question:', qa['question'])
    print('Ground Truth:', qa['answer'])
    print('Moondream:', moondream.answer_question(
        moondream.encode_image(sample['image']),
        qa['question'],
        tokenizer=tokenizer,
    ))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
moondream.text_model.train()
moondream.text_model.transformer.gradient_checkpointing_enable()

moondream.vision_encoder.train()


moondream.text_model.requires_grad = True
moondream.vision_encoder.requires_grad = True

teacher_moondream.text_model.requires_grad = False
teacher_moondream.vision_encoder.requires_grad = False


total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
optimizer = Adam8bit(
    [
        {"params": moondream.text_model.parameters()},
        {"params": moondream.vision_encoder.parameters()},
    ],
    lr=LR * 0.1,
    betas=(0.9, 0.95),
    eps=1e-6
)

if USE_WANDB:
    import wandb
    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        }
    )

#----------------------------------------Training loop with checkpoint saving-------------------------------------
i = 0
for epoch in range(EPOCHS):
    for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        i += 1

        loss,vision_loss = compute_loss(batch)
        loss.backward()

        if i % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Save checkpoint
        if i == 110 or i % 40 == 0:
            save_checkpoint(epoch, i, moondream.text_model, optimizer, loss.item(),"t")
            save_checkpoint(epoch, i, moondream.vision_encoder, optimizer, loss.item(),"v")


        if USE_WANDB:
            wandb.log({
                "loss/train": loss.item(),
                "loss/vision_distill": vision_loss,
                "lr": optimizer.param_groups[0]['lr']
            })

if USE_WANDB:
    wandb.finish()
