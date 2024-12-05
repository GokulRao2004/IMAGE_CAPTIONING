import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Specify the local path to the downloaded model
local_model_path = "D:\model\checkpoints\Llama3.2-11B-Vision"

# Load the model from the local directory
model = MllamaForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load the processor from the local directory
processor = AutoProcessor.from_pretrained(local_model_path)

# Example image to use
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define the messages as a list of dictionaries
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]

# Prepare the input text using the processor's chat template
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Process the image and the input text and move them to the model's device
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

# Generate the output with a limit on the number of new tokens
output = model.generate(**inputs, max_new_tokens=30)

# Decode and print the generated text
print(processor.decode(output[0]))
