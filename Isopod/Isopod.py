import numpy as np 
import cv2 as cv

def dummy_func(a):
    """Dummy function
    
    Returns nothing because it is a stupid function.
    
    """

    return None



class isopod:
    """
    Isopod Class

    This is a temporary class description.
    
    """


    def __init__(self, 
                 c_thr = 0.0003, 
                 e_thr= 10.,
                 sigma = 4.,
                 n_oct_layers = 8,
                 size_threshold = 2.,
                 edge_threshold = 1.,
                 path = None):
                 
        """Class initialization function

        Initializes Isopod class and sets some default class variables.
        
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

        """

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
        
        def get_image(self, *images):
            """
            Gets image(s) and adds it in color and grayscale to lists

            Parameters:
            ----------
            *images: str
                Variable number of image filenames to be opened.
            """

            for image in images:
                if self.path is not None:
                    opened_image = cv.imread(self.path+image)
                else:
                    opened_image = cv.imread(image)

                self.images.append(opened_image)
                self.grayscale_images.append(cv.cvtColor(opened_image, cv.COLOR_BGR2GRAY))
            
        
        
        def calculate_keypoints(self):
            """
            Applies SIFT algorithm??

            Parameters:
            ----------
            """

            #loop through grayscale images to apply sift to each image
            for gray_image in self.grayscale_images:

                #initialize sift instance
                self.sift = cv.SIFT.create(contrastThreshold = self.c_thr,
                                        edgeThreshold=self.e_thr,
                                        sigma=self.sigma,
                                        nOctaveLayers=self.n_oct_layers)
                
                #detect keypoints and compute descriptors
                keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
                self.keypoints.append(np.array(keypoints))
                self.descriptors.append(np.array(descriptors))
        
        def match_keypoints(self,k ):
            
            for g1, gray1 in enumerate(self.grayscale_images):
                for g2, gray2 in enumerate(self.grayscale_images):

                    if g1 != g2:
                        

            bf = cv.BFMatcher()
            matches = bf.knnMatch()

  