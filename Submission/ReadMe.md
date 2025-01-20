# ADOBE - Image Classification And Artifact Detection - Inter IIT Tech Meet 13.0

## Requirements
- Create a Python environment with Python version Python 3.10.12
- For training model for task 2 install torch 2.5.1 with Unsloth as dependency.
- Please install torch 2.2.0 with stable fast for fast generation of images.
- Install all the required packages using the following command -
  `pip install -r requirements.txt`


## Steps to run Inference script

The final Model was uploaded on hugging face.
- In the `inference_Pixtral.py` replace `HF_LOGIN_KEY` with Hugging face login key provided.
- Provide the Path to image for inference at `IMAGE_PATH`


## Steps to train model for task 1
- For training model for image classification (task 1) run the `train_task1.py` python file. 
- Provide your `WANDB_API_KEY` and `WANDB_PROJECT_NAME` in the training script.
- Replace the `primary_dataset_path` with dataset path.
- The `MODEL` should be replaced with the model backbone.
- For training the model with adverasrial defence run `train_task_1.py` python file.
- To Train model with subnetworks for better accuracy run `train_task1_subnet.py` file.


## Steps to train model for Task 2
- For training model for Artifact Detection (task 2) install the required packages from the requirements.txt file
- Run the `training_Pixtral.py` for training moondream for task 2. Use only the `HUGGING_FACE_LOGIN_TOKEN` provided the path is already provided in the file for dataset.

## Steps to run the Interface
- Create a Python environment with Python version 3.10.12
- Install all the required packages using the following command -
    `pip install gradio` 
- Give the Image to the model as input.
- After installations, to start the interface, run `main.py` using the command -

  `gradio main.py`

## Steps to run validation Script
- In the `validation_script.py` replace the `MODEL` with the backbone of the model.
- Replace the `CKPT_PATH` with the actual checkpoint of the path.
- Other model parameters can be changed according to the requirements.
- Validation sets will contain the path for the validation dataset . 

## Image generation Scripts
- Please install torch 2.2.0 with stable fast for fast generation of images.
- Run `image_gen.py` for generation images with SD_1.5 model.
- For image genaration from SDXL model and PixArt-Sigma-XL run `image_gen_SDXL.py` and `image_gen_Pixart.py` python file respectively.
- Replace the path of image directory with your path for saving the images.




