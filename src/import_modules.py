# Python modules
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from os import listdir
from sklearn.cluster import KMeans
import numpy as np
import pickle

# Our own files
from constants import N_IMAGES
from sift_functions import get_descriptor_matrix, get_descriptor_matrix_10_10
from helper_functions import get_all_categories
