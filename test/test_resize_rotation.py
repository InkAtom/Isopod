import sys
sys.path.append('/home/brainless/Master/COSMOS/Isopod/Isopod')


from Isopod import isopod

def test_match_keypoints():
    isp = isopod(n_oct_layers=10, sigma=2)
    isp.get_image("testimg1.png", "testimg2.png")
    isp.calculate_keypoints()
    isp.match_keypoints(0.5)
    isp.resize_images(0.5)
    isp.rotate_images(0.5)

    assert len(isp.matches)!=0