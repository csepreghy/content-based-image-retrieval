import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error

img = cv2.imread("object_categories/brain/image_0003.jpg",
                 cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
(keypoints, descriptors) = sift.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, None)

# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(img)
plt.show()

print(img)

print("kps: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
