# Python modules
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from os import listdir
from sklearn.cluster import KMeans
import numpy as np

# Our own files
from constants import N_IMAGES
from sift_functions import get_descriptor_matrix
from helper_functions import get_all_categories
