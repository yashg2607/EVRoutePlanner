from PIL import Image

# Load the image
image_path = '../../../../../../../../Desktop/image_67500.tiff'
image = Image.open(image_path)

# Set the scaling factor
scaling_factor = 10

# Calculate the new dimensions
width, height = image.size
new_width = width * scaling_factor
new_height = height * scaling_factor

# Upscale the image using nearest-neighbor interpolation
upscaled_image = image.resize((new_width, new_height), Image.NEAREST)

# Save the upscaled image
upscaled_image.save('upscaled_image1.png')