# Python modules
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from os import listdir
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pickle
import pandas as pd
from scipy.spatial import distance

# Our own files
from constants import N_IMAGES, k
from image_processing_functions import *
from helper_functions import get_all_categories
