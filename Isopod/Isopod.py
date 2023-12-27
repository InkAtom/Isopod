import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import imutils

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


    def resize_images(self, match_distance = 0.1):
        """Image resizing
        
        This method resizes the larger image to be the same as the smaller one 
        in terms of keypoint size. Match_distance might need to be made larger since interpolating
        leads to some local errors.

        Args: 
            match_distance (float): Parameter for the isopod.match_keypoints function
        
        """

        #initialize array holding the size difference
        size_differences = []

        #compare sizes and append result to size array
        for match in self.matches:
            size1 = self.keypoints[0][match[0].queryIdx].size
            size2 = self.keypoints[1][match[0].trainIdx].size

            size_difference = size1/size2
            size_differences.append(size_difference) 
        
        #exit function if the size difference is very small
        if np.max(np.abs(size_differences))<0.01:
            return
        
        #get the mean amount to resize by (not all size differences are the same due to errors)
        resize_amount = np.mean(size_differences)
        
        #decide which image to resize
        if resize_amount>1:
            which = 0
        else:
            which = 1

        #take absolute value of the resize amount (to not shrink it in case 2)
        resize_amount=np.abs(resize_amount)

        #resize the first or second image, depending on which is larger
        self.grayscale_images[which] = cv.resize(self.grayscale_images[which],
                                                dsize=(int(np.shape(self.grayscale_images[which])[0]*resize_amount),
                                                    int(np.shape(self.grayscale_images[which])[1]*resize_amount)),
                                                interpolation = cv.INTER_AREA)
        
        #redo keypoint caculation and matching
        keypoints, descriptors = self.sift.detectAndCompute(self.grayscale_images[which], None)
        self.keypoints[which] = np.array(keypoints)
        self.descriptors[which] = np.array(descriptors)

        #get new matches
        self.match_keypoints(match_distance)                                              





    def rotate_images(self, match_distance=0.1):
        """Match descriptor orientation comparison

        Extracts the difference in orientation and rotates one of the images so that both are
        oriented the same way. The rotated image has its keypoints extracted and matched again
        for further match distance calculation.

        Args: 
            match_distance (float): Parameter for the isopod.match_keypoints function
        
        """
        
        #initialize list to hold orientation differences
        orientation_differences = []

        #go through each match and compare the orientations of the matched keypoints
        for match in self.matches:
            orientation1 = self.keypoints[0][match[0].queryIdx].angle
            orientation2 = self.keypoints[1][match[0].trainIdx].angle

            orientation_difference = orientation1-orientation2
            orientation_differences.append(orientation_difference)
        
        #find direction
        if orientation_differences[np.argmin(np.abs(orientation_differences))]>=0:
            direction = 1
        else:
            direction = -1

        #get mean orientation difference as the smalles angle
        # and express it as the rotation needed in degrees
        for i, _ in enumerate(orientation_differences):
            orientation_differences[i] = np.min([np.abs(orientation_differences[i]), 
                                                360-np.abs(orientation_differences[i])])
        
        rotation_angle = np.mean(orientation_differences)

        #pad the first image with zeroes so that rotation does not lead to any data loss
        #get the difference between side lengths and diagonal to know how much to pad
        diagonal = int(np.sqrt(np.sum(np.array(np.shape(self.grayscale_images[0]))**2)))
        xdiff = int((diagonal - np.shape(self.grayscale_images[0])[0])/2)
        ydiff = int((diagonal - np.shape(self.grayscale_images[0])[1])/2)
        self.grayscale_images[0] = np.pad(self.grayscale_images[0], 
                                          ((xdiff,xdiff),(ydiff,ydiff)))
        
        #rotate image
        self.grayscale_images[0] = imutils.rotate(self.grayscale_images[0], 
                                                  angle = direction*rotation_angle)
        
        #calculate keypoints and descriptors again for image 1
        keypoints, descriptors = self.sift.detectAndCompute(self.grayscale_images[0], None)
        self.keypoints[0] = np.array(keypoints)
        self.descriptors[0] = np.array(descriptors)

        #get new matches
        self.match_keypoints(match_distance)

    def stitch_images(self):
        '''Image stitching method

        This method stitches the two images together to create one final image. This is done
        by figuring out the final image size and then "implanting" the images at the correct positions

        Returns:
            final_canvas (array): The final stitched image created by putting the two given images 
                at the correct positions on a blank canvas. Open spaces remain zero.
        '''

        #we can take any match to figure out the dimensions, for ease the first one is taken
        match0 = self.matches[0][0]

        #get positions of each image's point
        positions = [self.keypoints[0][match0.queryIdx].pt,
                     self.keypoints[1][match0.trainIdx].pt]
        
        #get image distances in x and y from each side
        distance0 = [positions[0][0], 
                     positions[0][1],
                     np.abs(np.shape(self.grayscale_images[0])[1]-positions[0][0]),
                     np.abs(np.shape(self.grayscale_images[0])[0]-positions[0][1])]
        distance1 = [positions[1][0], 
                     positions[1][1],
                     np.abs(np.shape(self.grayscale_images[1])[1]-positions[1][0]),
                     np.abs(np.shape(self.grayscale_images[1])[0]-positions[1][1])]

        #initialize list holding maximum distance to border from the point and then append the max distances
        max_distance = []
        max_distance.append(np.max([distance0[0],distance1[0]]))
        max_distance.append(np.max([distance0[1],distance1[1]]))
        max_distance.append(np.max([distance0[2],distance1[2]]))
        max_distance.append(np.max([distance0[3],distance1[3]]))
        
        #for easier final image placement, also get which image led to the max border
        max_distanceidx = []
        max_distanceidx.append(np.argmax([distance0[0],distance1[0]]))
        max_distanceidx.append(np.argmax([distance0[1],distance1[1]]))
        max_distanceidx.append(np.argmax([distance0[2],distance1[2]]))
        max_distanceidx.append(np.argmax([distance0[3],distance1[3]]))

        #get the final image max size and make a zerofilled matrix with it
        final_canvas = np.zeros(shape = (int(max_distance[1]+max_distance[3]),
                                         int(max_distance[0]+max_distance[2])))
        
        #find offsets
        max_distance = np.array(max_distance)
        max_distanceidx = np.array(max_distanceidx)
        distance0 = np.array(distance0)
        distance1 = np.array(distance1)
        image0_offsets = np.floor(np.where(max_distanceidx ==1, max_distance-distance0, 0)).astype(int)
        image1_offsets = np.floor(np.where(max_distanceidx ==0, max_distance-distance1, 0)).astype(int)

        #put onto final canvas        
        #insert first image
        final_canvas[image0_offsets[1]:image0_offsets[1]+np.shape(self.grayscale_images[0])[0], 
                image0_offsets[0]:image0_offsets[0]+np.shape(self.grayscale_images[0])[1]] = self.grayscale_images[0]


        #get maximum values (i.e. compute overlap) and add to final figure
        overlap_image1 = np.amax([final_canvas[image1_offsets[1]:image1_offsets[1]+np.shape(self.grayscale_images[1])[0], 
                     image1_offsets[0]:image1_offsets[0]+np.shape(self.grayscale_images[1])[1]], self.grayscale_images[1]],axis=0)
    
        final_canvas[image1_offsets[1]:image1_offsets[1]+np.shape(self.grayscale_images[1])[0], 
                     image1_offsets[0]:image1_offsets[0]+np.shape(self.grayscale_images[1])[1]] = overlap_image1
              
        #return stitched image
        return final_canvas
        

                            

        

        

if __name__ == "__main__":
    
    isp = isopod()
    isp.get_image("cut_1.png", "cut_2.png")
    isp.calculate_keypoints()
    isp.match_keypoints(0.1)
    
    new_img = cv.drawMatchesKnn(isp.grayscale_images[0], isp.keypoints[0],
                               isp.grayscale_images[1], isp.keypoints[1],
                               isp.matches, None,
                               flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    isp.resize_images(0.1)
    isp.rotate_images(0.1)

    new_img2 = cv.drawMatchesKnn(isp.grayscale_images[0], isp.keypoints[0],
                               isp.grayscale_images[1], isp.keypoints[1],
                               isp.matches, None,
                               flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    final_image = isp.stitch_images()

    fig,ax = plt.subplots(2)
    ax[1].imshow(final_image)
    ax[0].imshow(new_img2)
    plt.show()