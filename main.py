import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error

IMG = cv2.imread("object_categories/brain/image_0003.jpg",
                 cv2.IMREAD_GRAYSCALE)

SIFT = cv2.xfeatures2d.SIFT_create()
(KEYPOINTS, DESCRIPTORS) = SIFT.detectAndCompute(IMG, None)

IMG = cv2.drawKeypoints(IMG, KEYPOINTS, None)

cv2.imshow("Image", IMG)
cv2.waitKey(0)

cv2.destroyAllWindows()

print(IMG)

print("kps: {}, descriptors: {}".format(len(KEYPOINTS), DESCRIPTORS.shape))
