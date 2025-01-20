import gradio as gr
import json
from Pipeline import *
"""
This is the Integrated UI file made with Gradio
"""

IMAGE_PATH = "image.png" # add path_to_image

def predict_image(image):
    if image is None:
        return "No image uploaded", gr.update(visible=False), gr.update(visible=False)
    
    
    
    prediction, res2 = process_prediction(image)
    if prediction == "FAKE":
        
        return prediction, gr.update(visible=True, value=json.dumps(res2, indent=2)), gr.update(visible=True)
    else: 
       
        return prediction, gr.update(visible=False), gr.update(visible=True)
    
   
def remove_image():
    return None, "Image removed", gr.update(visible=False), gr.update(visible=False)

# Gradio interface
def main():
    with gr.Blocks() as demo:
        with gr.Column(elem_id="centered-column"):
            gr.Markdown("# Real vs AI Face Detection", elem_id="title")
            gr.Markdown("Upload an image to see if it's a real or AI-generated one.", elem_id="subtitle")
            
            image_input = gr.Image(type="pil", label="Choose an Image", elem_id="image-input")
            
            # Buttons placed side by side
            with gr.Row(elem_id="button-row"):
                predict_button = gr.Button("Predict Result", elem_id="predict-button")
                remove_button = gr.Button("Remove Image", elem_id="remove-button", visible=False)  # Initially hidden
            
            prediction_text = gr.Textbox(label="Prediction Results", interactive=False)
            json_output = gr.Textbox(label="Details (JSON)", interactive=False, visible=False)  # Initially hidden
            
            # Predict button functionality
            predict_button.click(
                predict_image,
                inputs=[image_input],
                outputs=[prediction_text, json_output, remove_button]
            )
            
            # Remove button functionality
            remove_button.click(
                remove_image,
                inputs=None,
                outputs=[image_input, prediction_text, json_output, remove_button]
            )

    # Add custom CSS
    demo.css = """
        #centered-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh; /* Full viewport height for vertical centering */
        }
        #title, #subtitle {
            text-align: center;
        }
        #image-input {
            width: 500px; /* Reduced input box width */
        }
        #button-row {
            display: flex;
            justify-content: center;
            gap: 10px; /* Space between buttons */
            margin-top: 10px;
        }
        #predict-button {
            background-color: #4CAF50; /* Green color */
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }
        #predict-button:hover {
            background-color: #45a049; /* Slightly darker green on hover */
        }
        #remove-button {
            background-color: #f44336; /* Red color */
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }
        #remove-button:hover {
            background-color: #e53935; /* Slightly darker red on hover */
        }
    """
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()
