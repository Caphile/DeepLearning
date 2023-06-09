import scipy.io as io
import numpy as np

# load the data
data = io.loadmat("face_landmark.mat")
images = data["images"]
landmarks = data["landmarks"]
print("im_shape:", images.shape)
print("landmarks_shape:", landmarks.shape)

# visualize a random data
np.random.seed(101)
id = np.random.randint(len(images))
im = images[id]
keypoints = landmarks[id]

from matplotlib import pyplot as plt
plt.imshow(im, cmap="gray")
for point in keypoints:
    plt.plot(point[0], point[1], "r+")
plt.show()
