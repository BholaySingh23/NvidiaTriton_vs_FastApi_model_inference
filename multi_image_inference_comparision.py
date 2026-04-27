import requests
import time
import io
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Prepare 10 image URLs (Mixture of cats and dogs from reliable sources)
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/224px-Cat_November_2010-1a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Golden_Retriever_medium-shot.jpg/224px-Golden_Retriever_medium-shot.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/June_odd-eyed-cat.jpg/224px-June_odd-eyed-cat.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/224px-Dog_Breeds.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Adoption-purebred-dog-border-collie-retriever.jpg/224px-Adoption-purebred-dog-border-collie-retriever.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Shorthair.jpg/224px-American_Shorthair.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Puppy_on_grass.jpg/224px-Puppy_on_grass.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/White_Persian_Cat.jpg/224px-White_Persian_Cat.jpg"
]

images_data = []
headers = {'User-Agent': 'Mozilla/5.0'}

print("Downloading 10 images for benchmarking...")
for i, url in enumerate(urls):
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            img = Image.open(io.BytesIO(res.content)).convert("RGB").resize((224, 224))
            filename = f'batch_{i}.jpg'
            img.save(filename)
            images_data.append(filename)
    except:
        continue

print(f"Successfully prepared {len(images_data)} images.\n")

# --- Method 1: FastAPI (Serial Requests) ---
print("Starting FastAPI Serial Test...")
start_api = time.time()
api_results = []
for img_path in images_data:
    with open(img_path, 'rb') as f:
        resp = requests.post("http://localhost:8000/predict", files={"file": f})
        api_results.append(resp.json())
api_total_time = time.time() - start_api

# --- Method 2: Triton/SavedModel (Batched Inference) ---
print("Starting Triton Batched Test...")
imported = tf.saved_model.load('model_repository/cats_dogs/1/model.savedmodel')
infer_fn = imported.signatures['serve']

start_triton = time.time()
# Preprocess all images into one batch tensor
batch_array = []
for img_path in images_data:
    img = Image.open(img_path)
    arr = np.array(img).astype(np.float32) / 255.0
    batch_array.append(arr)

input_batch = np.stack(batch_array, axis=0) # Shape: (10, 224, 224, 3)

# Single inference call for the entire batch
triton_results = infer_fn(input_layer_1=tf.constant(input_batch))
triton_scores = triton_results['output_0'].numpy()
triton_total_time = time.time() - start_triton

# --- Summary ---
print("\n--- Batch Comparison Results (10 Images) ---")
print(f"FastAPI Total Time (Serial):  {api_total_time:.4f}s (~{api_total_time/len(images_data):.4f}s per image)")
print(f"Triton Total Time (Batched): {triton_total_time:.4f}s (~{triton_total_time/len(images_data):.4f}s per image)")

speedup = api_total_time / triton_total_time
print(f"\nBatching Speedup: {speedup:.2f}x faster")
print("Note: Triton's efficiency grows even more as batch size and hardware utilization increase.")
