from PIL import Image
from IPython.display import display
try:
    # Open the image file from the specified path
    img = Image.open('/content/real_test.jpg')
    
    # Display the image using your operating system's default viewer
    display(img)
    
    # Optional: print basic image metadata
    print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
    
except IOError:
    print("Unable to load image. Check the file path or format.")
