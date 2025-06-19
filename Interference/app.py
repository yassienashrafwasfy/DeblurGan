import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Normalize: [0, 255] → [-1, 1]
def normalize(img):
    return img / 127.5 - 1.0

# Denormalize: [-1, 1] → [0, 255]
def denormalize(img):
    return ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

# Load the model
model = tf.keras.models.load_model("Model Weights/improved_deblur_gan_epoch_15.h5", compile=False)

# Enhanced prediction function that returns multiple images
def predict_and_display(input_img):
    if input_img is None:
        return None, None, None
    
    # Convert to PIL Image if it's a numpy array
    if isinstance(input_img, np.ndarray):
        original_img = Image.fromarray(input_img.astype(np.uint8))
    else:
        original_img = input_img
    
    # Resize to model's expected input size (256x256)
    resized_img = tf.image.resize(input_img, (256, 256), method='area')
    resized_img_pil = Image.fromarray(resized_img.numpy().astype(np.uint8))
    
    # Normalize and predict
    img_normalized = normalize(np.array(resized_img))
    img_batch = np.expand_dims(img_normalized, axis=0)
    pred = model.predict(img_batch)[0]
    pred_img = denormalize(pred)
    pred_img_pil = Image.fromarray(pred_img)
    
    return original_img, resized_img_pil, pred_img_pil

# Create the interface using Blocks for better layout control
with gr.Blocks(title="Deblur GAN - Enhanced View") as demo:
    gr.Markdown("# Deblur GAN")
    gr.Markdown("Upload a blurry image to see the deblurred result. View the original image, resized input (256×256), and the AI-enhanced output side by side.")
    
    with gr.Row():
        input_image = gr.Image(label="Upload Blurry Image", type="numpy")
    
    with gr.Row():
        original_display = gr.Image(label="Original Input", interactive=False)
        resized_display = gr.Image(label="Resized Input (256×256)", interactive=False)
        output_display = gr.Image(label="Deblurred Output", interactive=False)
    
    # Process button
    process_btn = gr.Button("Process Image", variant="primary")
    
    # Connect the function to the button only
    process_btn.click(
        fn=predict_and_display,
        inputs=input_image,
        outputs=[original_display, resized_display, output_display]
    )

demo.launch()