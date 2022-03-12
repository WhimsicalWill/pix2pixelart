import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("harbor.jpg")
pixart = np.ones_like(image)

# iterate through each 3x3 block of the image and take an average
block_size = 6
height, width = image.shape[0], image.shape[1]

# outer loop, loop through the width
for x in range(width // block_size):
    for y in range(height // block_size):
        sx, sy = block_size*x, block_size*y # the coordinates starting the blocks for x, y
        avg_pix = [np.mean(image[sy:sy+block_size, sx:sx+block_size, c]) for c in range(3)]
        for i, color_avg in enumerate(avg_pix): # each color has separate avg to avoid grayscale
            pixart[sy:sy+block_size, sx:sx+block_size, i] *= int(color_avg) # cast to int for matplotlib image

fig, axs = plt.subplots(2, 1)
fig.suptitle("Original image vs. naive pixelart image")
axs[0].imshow(image)
axs[1].imshow(pixart)
plt.show()
