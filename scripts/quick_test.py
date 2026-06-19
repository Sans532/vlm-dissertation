import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen25vl-7b"

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

# Create a simple test image (red square on white background)
img = Image.new("RGB", (224, 224), "white")
for x in range(50, 174):
    for y in range(50, 174):
        img.putpixel((x, y), (255, 0, 0))
img.save(f"/home/{USER}/dissertation/data/test_image.png")
print("Test image created.")

# Ask model about the image
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": f"file:///home/{USER}/dissertation/data/test_image.png"},
        {"type": "text", "text": "What do you see in this image? Describe it."}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
from qwen_vl_utils import process_vision_info
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_new_tokens=100)
answer = processor.batch_decode(out, skip_special_tokens=True)[0]

print(f"\nQuestion: What do you see in this image?")
print(f"Answer: {answer}")
print("\n=== MODEL IS WORKING ===")
