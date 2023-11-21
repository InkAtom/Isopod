import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

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
    
    def match_keypoints(self,match_distance=0.1):

        #list of matches to associated images
        self.matches = []

        bf = cv.BFMatcher() 
        matches = bf.knnMatch(self.descriptors[0],self.descriptors[1],k=2)

        for m,n in matches:
            if m.distance<match_distance*n.distance:
                self.matches.append([m])
        

if __name__ == "__main__":
    
    isp = isopod()
    isp.get_image("cut_1.png", "cut_2.png")
    isp.calculate_keypoints()
    isp.match_keypoints(0.05)

    
    new_img = cv.drawMatchesKnn(isp.grayscale_images[0], isp.keypoints[0],
                               isp.grayscale_images[1], isp.keypoints[1],
                               isp.matches_list[0][2], None,
                               flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(new_img)
    plt.show()