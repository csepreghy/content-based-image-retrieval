from constants import NUM_OF_IMAGES

import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

descriptor_matrix = []

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error

sift = cv2.xfeatures2d.SIFT_create()

for i in range(NUM_OF_IMAGES):
  print(i)
  if i < 10:
    img = cv2.imread("object_categories/brain/image_000" +
                     str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

  elif i >= 10 & i < 100:
    img = cv2.imread("object_categories/brain/image_00" +
                     str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

  elif i >= 100 & i < 1000:
    img = cv2.imread("object_categories/brain/image_0" +
                     str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

  elif i >= 100 & i < 10000:
    img = cv2.imread("object_categories/brain/image_" +
                     str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)

  keypoints, descriptors = sift.detectAndCompute(img, None)
  img = cv2.drawKeypoints(img, keypoints, None)

  descriptor_matrix.append(descriptors)

  plt.imshow(img)
  plt.show()
  # cv2.imshow("Image", img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()


print("kps: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
