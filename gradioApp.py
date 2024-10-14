import gradio as gr
from PIL import Image
import io
from infer import ImageCaptioningInference
from models.model import ImageCaptioningModel
import numpy as np

# Initialize the model
model_dir = 'model'
model = ImageCaptioningModel()
model.load(model_dir)
inference_model = ImageCaptioningInference(model)

def generate_caption(image):
    if image is None:
        return "No image provided."
    
    try:
        # Generate caption using the image path
        generated_caption = inference_model.infer_image(image)
        return generated_caption
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning",
    description="Upload an image or select one from your folder to generate a caption.",
    examples=[["test_img.jpg"]]  # Add some example images if available
)

# Launch the app
if __name__ == "__main__":
    iface.launch()