## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

## AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

## PROBLEM STATEMENT:
Image captioning is the task of generating a textual description for a given image. This is crucial for applications like accessibility tools for visually impaired users, content generation, and image indexing. Traditional systems often rely on predefined labels or are limited in context understanding. By leveraging the BLIP model—a state-of-the-art vision-language pretraining model—this project aims to create an intuitive and efficient application for real-time image captioning, accessible via the Gradio interface.

## DESIGN STEPS:
## STEP 1:
Model Preparation
Use a pre-trained BLIP image-captioning model available from Hugging Face Transformers or similar libraries. Ensure the model supports inference on diverse image types and contexts.

## STEP 2:
Framework
Use Gradio to create a UI with the following components: Input: File upload for images. Output: Textbox showing the generated caption.

## STEP 3:
Workflow
Load the BLIP model and tokenizer. Accept an image as input via Gradio's file upload. Preprocess the image for the BLIP model. Generate a caption using the BLIP model's inference pipeline. Display the caption on the Gradio interface.

## STEP 4:
Testing and Deployment
Test the application with various image types to ensure the captions are meaningful and diverse. Deploy the application on a public URL using Gradio’s hosting features or external platforms like Hugging Face Spaces.
### PROGRAM:
```
import os
import io
import IPython.display
from PIL import Image
import base64
from dotenv import load_dotenv, find_dotenv
import requests
import json
import gradio as gr

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Access environment variables
hf_api_key = "your_huggingface_api_key_here"
ENDPOINT_URL = "https://api-inference.huggingface.co/models/your-model-id"
PORT = 7860 
PORT = int(os.environ['PORT1'])                # Port for Gradio app

# Function to call Hugging Face API for image captioning
def get_completion(inputs, parameters=None):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json",
    }
    data = {"inputs": inputs}
    if parameters:
        data["parameters"] = parameters
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Error {response.status_code}: {response.text}")

# Convert PIL image to base64 string
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format="PNG")
    byte_arr = byte_arr.getvalue()
    return base64.b64encode(byte_arr).decode("utf-8")

# Generate caption from image
def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)
    return result[0]["generated_text"]

# Display sample image
image_url = "https://free-images.com/sm/9596/dog_animal_greyhound_983023.jpg"
IPython.display.display(IPython.display.Image(url=image_url))

# Create Gradio interface
gr.close_all()  # Ensure no previous Gradio instances are running

demo = gr.Interface(
    fn=captioner,
    inputs=[gr.Image(label="Upload image", type="pil")],
    outputs=[gr.Textbox(label="Caption")],
    title="Image Captioning with BLIP",
    description="Caption any image using the BLIP model.",
    allow_flagging="never",
    examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"],
)

# Launch app
demo.launch(share=True, server_port=PORT)
demo.close()

```

### OUTPUT:
Example Input and Output

Input:


![image](https://github.com/user-attachments/assets/c87f74f5-6922-495d-aebe-56cc20413c47)





Output:

there is a dog wearing a red Christmas hat and scarf


### RESULT:
The application successfully generates high-quality images based on user-provided text prompts. The Stable Diffusion model ensures visually appealing results, and the Gradio interface makes it accessible and interactive.
