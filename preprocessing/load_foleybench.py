from datasets import load_dataset
import base64

# Load the dataset
dataset = load_dataset("FoleyBench/foleybench")

# Access video data
first_example = dataset["train"][0]
print(f"Caption: {first_example['caption']}")
print(f"Duration: {first_example['duration']} seconds")

# Decode video
video_b64 = first_example["video_data"]
video_bytes = base64.b64decode(video_b64)

# Save video to file
with open("video.mp4", "wb") as f:
    f.write(video_bytes)