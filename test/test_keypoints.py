import cv2 as cv


from Isopod.Isopod import isopod

def test_find_keypoints():
    isp = isopod(n_oct_layers=10, sigma=2)
    isp.get_image("testimg1.png", "testimg2.png")
    isp.calculate_keypoints()

    assert len(isp.keypoints[0])!=0
    assert len(isp.keypoints[1])!=0
    assert type(isp.keypoints[0][0]) == cv.KeyPoint

