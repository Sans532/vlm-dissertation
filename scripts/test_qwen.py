import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

# Path to downloaded model
USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen2vl"

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully.")

def ask_qwen(video_path, question):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": 1.0
            },
            {"type": "text", "text": question}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to("cuda")

    out = model.generate(**inputs, max_new_tokens=200)
    return processor.batch_decode(out, skip_special_tokens=True)[0]

# ============ TEST ============
# Replace with any test video path
TEST_VIDEO = f"/home/{USER}/dissertation/data/test_video.mp4"

QUESTION = "What is the skill level of the person in this video? Answer only one: Novice / Early Expert / Intermediate Expert / Late Expert"

if os.path.exists(TEST_VIDEO):
    answer = ask_qwen(TEST_VIDEO, QUESTION)
    print(f"\nQuestion: {QUESTION}")
    print(f"Answer: {answer}")
else:
    print(f"No test video found at {TEST_VIDEO}")
    print("Model loaded fine. Ready to run when you add videos.")
