import numpy as np
import cv2
import matplotlib.pyplot as plt

# Reading the image
img = cv2.imread('image.jpg')
# Converting to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

resize_dim = 500
# Resizing the image
resized_img = cv2.resize(gray_img, (resize_dim, resize_dim))

# Applying SVD
U, S, V = np.linalg.svd(resized_img, full_matrices=False)

# low rank approximation, applying the approximation formula directly on the SVD components
# instead of using for loops to ensure faster execution
def low_rank_approximation(A, k):
    U, S, V = np.linalg.svd(A)
    A_reconstructed = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return A_reconstructed



plt.figure(figsize=(15, 6))
count = 0;
for k in range(1, min(resize_dim, resize_dim) + 1, 10):
    approximate_image = low_rank_approximation(resized_img, k)

    # Plotting
    plt.subplot(3, 4, count + 1) # subplot with size (width 4, height 3)
    plt.imshow(approximate_image, cmap="gray")
    plt.title(f"k = {k}")
    plt.axis("off")
    count += 1
    if count == 12:
        break

plt.tight_layout()
plt.show()


#  the lowest k such that I can clearly read out the author’s name from the image is 35