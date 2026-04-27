import requests
import time
import io
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Setup Data
cat_url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
headers = {'User-Agent': 'Mozilla/5.0'}
img_response = requests.get(cat_url, headers=headers)
img_raw = Image.open(io.BytesIO(img_response.content))
img_raw.save('benchmark_cat.jpg')

print(f"--- Benchmarking Image Classification Methods ---\n")

# --- Method 1: FastAPI Benchmark ---
start_api = time.time()
try:
    with open('benchmark_cat.jpg', 'rb') as f:
        api_resp = requests.post("http://localhost:8000/predict", files={"file": f})
    api_result = api_resp.json()
    api_time = time.time() - start_api
    print(f"[FastAPI] Label: {api_result['label']}, Score: {api_result['confidence']:.4f}, Time: {api_time:.4f}s")
except Exception as e:
    print(f"FastAPI Error: {e}")
    api_time = None

# --- Method 2: Triton-Style (SavedModel) Benchmark ---
# Note: Triton usually performs preprocessing on the client or via a pipeline.
# We include the load/signature step to reflect the backend logic.
start_triton = time.time()
imported = tf.saved_model.load('model_repository/cats_dogs/1/model.savedmodel')
infer_fn = imported.signatures['serve']

# Preprocessing
img = Image.open('benchmark_cat.jpg').convert("RGB").resize((224, 224))
img_array = np.array(img).astype(np.float32) / 255.0
input_tensor = np.expand_dims(img_array, axis=0)

# Inference
triton_results = infer_fn(input_layer_1=tf.constant(input_tensor))
raw_score = triton_results['output_0'].numpy()[0][0]
triton_label = "dog" if raw_score > 0.5 else "cat"
triton_time = time.time() - start_triton

print(f"[Triton Sim] Label: {triton_label}, Score: {raw_score:.4f}, Time: {triton_time:.4f}s")

# --- Final Comparison ---
print("\n--- Analysis ---")
if api_time and triton_time:
    diff = abs(api_time - triton_time)
    fastest = "FastAPI" if api_time < triton_time else "Triton (SavedModel)"
    print(f"Fastest Method: {fastest}")
    print(f"Time Difference: {diff:.4f}s")
    print("Efficiency Note: FastAPI includes HTTP overhead and Python GIL limits.")
    print("Triton's real-world advantage appears with concurrent requests and batching,")
    print("which are not fully captured in a single serial execution test.")
