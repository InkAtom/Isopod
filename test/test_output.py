from Isopod.Isopod import isopod
import numpy as np

def test_match_keypoints():
    isp = isopod(n_oct_layers=10, sigma=2)
    isp.get_image("testimg1.png", "testimg2.png")
    isp.calculate_keypoints()
    isp.match_keypoints(0.5)
    isp.resize_images(0.5)
    isp.rotate_images(0.5)
    img = isp.stitch_images()

    #test that an overlap is happening 
    assert np.shape(img)[0] < (np.shape(isp.grayscale_images[0])[0] + np.shape(isp.grayscale_images[1])[0])
    assert np.shape(img)[1] < (np.shape(isp.grayscale_images[0])[1] + np.shape(isp.grayscale_images[1])[1])