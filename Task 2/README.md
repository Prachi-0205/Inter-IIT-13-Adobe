# Task 2 
To list-out artifacts in the 32X32 AI generated Image from the given list of Artifact. And to explain the reason for the artifacts detected in the 32X32 Image. <br>
# Final Approach Used 
## Dataset Generation 
* Images were created based on the CIKAFE template from SDXL,SD 1.5 and PixArt models which were originally 512X512. <br>
* The captioning was done using Gemini-1.5 Flash API to list out artifacts and explain them on the original 512X512 Images. <br>

[Dataset](https://huggingface.co/datasets/22-24/Final)  <br>

## Training 
* The Dataset was loaded from Hugging face and the Images were rescaled to 32X32 and then again rescaled to 512X512.  <br>
* The Pixtral 12-B Vision Language Model was trained using [Unsloth](https://github.com/unslothai/unsloth) on a NVIDIA A40 GPU and uploaded to Huggingface. <br>

[Model on Hugging Face](https://huggingface.co/22-24/pixtral_2) <br>

## Inference
* The trained model was loaded from hub and then tested for rescaled images from 32X32 to 512X512.<br>