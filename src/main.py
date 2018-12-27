from import_modules import *

style.use('fivethirtyeight') # for matplotlib

descriptor_matrix = get_descriptor_matrix(NUM_OF_IMAGES, img_category = "buddha")
# didn't make brain into a constant because we might have a list instead to loop througj

print(descriptor_matrix)

# If you get an error regarding cv2.xfeatures2d it's because in the new version that algorithm isn't free
# Do pip install opencv-contrib-python==3.4.1.15 to get rid of the error
