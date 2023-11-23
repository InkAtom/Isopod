import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

class isopod:
    """InterStitching Of Pictures based On Descriptors aka ISOPOD class

    Calling an instance of this class and feeding two images into it
    allows to automatically stitch them together, assuming they contain the  
    same area or object. The automatic detection of which areas are to be stitched together is
    based on the `SIFT`_ algorithm.

    Args:
            c_thr (float): Contrast threshold parameter for the SIFT algorithm. Defaults to 0.0003.
            e_thr (float): Edge threshold parameter for the SIFT algorithm. Defaults to 10.
            n_oct_layers (int): Parameter indicating the layers for each octave for the SIFt algorithm.
                Defaults to 8.
           path (str): If images can be found at a different location, the path to their folder should 
                be specified here. Defaults to None.

    .. _SIFT:
        https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
    """


    def __init__(self, 
                 c_thr = 0.0003, 
                 e_thr= 10.,
                 sigma = 4.,
                 n_oct_layers = 8,
                 path = None):
        """Initialization function for the isopod class

        This function sets up the default instance initialization for the isopod class. It
        also sets some instance variables, empty lists that will be filled. See class docstring.
                    
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
        #: list of array: List holding the original images.
        self.images = []

        #: list of array: List holding grayscale versions of the original images.
        self.grayscale_images = []

        #lists for keypoitns and descriptors
        #: list of cv.Keypoint: List holding the keypoints for the analyzed images.
        self.keypoints = []

        #: list of array: List holding the descriptors belonging to the keypoints in 
        self.descriptors = []
        
    def get_image(self, *images):
        """Image getter
        
        Gets images and adds them in color and grayscale to isopod.images.

        Args:
            *images: Variable number of image filenames to be opened. More than 2 can be opened,
                but only the first two will be stitched together. If images are in different folders,
                this function can be called once for each.
        
        """

        for image in images:
            if self.path is not None:
                opened_image = cv.imread(self.path+image)
            else:
                opened_image = cv.imread(image)

            self.images.append(opened_image)
            self.grayscale_images.append(cv.cvtColor(opened_image, cv.COLOR_BGR2GRAY))
        
    
    
    def calculate_keypoints(self):
        """Keypoint and Descriptor calculation
        
        This class function applies the SIFT algorithm to each image and calculates their keypoints
        and accompanying descriptors.
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
        """Keypoint matcher

        This function forms keypoint pairs between the images based on the distance between descriptors.

        Args:
            match_distance (float): This parameter regulates the acceptance threshold for descriptor pairs.
                L2-distances for the 2 nearest neighbours in image 2 are calculated for each descriptor
                in image 1. If the nearer neighbour is closer than match_distance*distance to second nearest 
                neighbour, then the descriptors are accepted as pair. Lower threshold leads to more accurate pairs.
                Defaults to 0.1.
        """

        #list of matches to associated images
        #: list of lists of cv.DMatch: List holding all the keypoint pairs. Shaped this way to be used with cv.drawMatchesKnn
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
    print(np.shape(isp.matches))
    print(type(isp.matches[0][0]))
    
    new_img = cv.drawMatchesKnn(isp.grayscale_images[0], isp.keypoints[0],
                               isp.grayscale_images[1], isp.keypoints[1],
                               isp.matches, None,
                               flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(new_img)
    plt.show()