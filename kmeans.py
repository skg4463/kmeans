import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("fruit_color.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

K = 3

print("pixels :", pixel_values.shape[0])
print("Cluster: ", K)

_, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

labels = labels.flatten()

segmented_image = centers[labels.flatten()]

for i in range(K):
    print("-------\nClass", i)
    print(round((labels == i).sum()/pixel_values.shape[0], 4)*100, "%")

print(pixel_values[0])

segmented_image = segmented_image.reshape(image.shape)
plt.imshow(segmented_image)
plt.show()
