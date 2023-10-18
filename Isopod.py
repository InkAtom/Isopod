import numpy as np 
import cv2 as cv

class isopod:

    def __init__(self, 
                 c_thr = 0.0003, 
                 e_thr= 10.,
                 sigma = 4.,
                 n_oct_layers = 8,
                 size_threshold = 2.,
                 edge_threshold = 1.,
                 path = None):
                 
        '''Class initialization function
        
        Parameters
        ----------
        c_thr : float, optional
            Contrast threshold for keypoint detection.
        
        e_thr : float, optional
            Edge threshold for keypoint detection.
        
        sigma : float, optional
            Blur factor applied at each octave.

        n_oct_layers : int, optional
            Number of layers per octave.
        
        path : str, optional
            System path to images. 

        '''

        #set parameter defaults for the class
        #SIFT parameters
        self.c_thr = c_thr
        self.e_thr = e_thr
        self.sigma = sigma
        self.n_oct_layers = n_oct_layers

        #other
        self.path = path

        #initialize image list
        self.images = []
        self.grayscale_images = []

        #lists for keypoitns and descriptors
        self.keypoints = []
        self.descriptors = []
        self.test = []  #git test thingy