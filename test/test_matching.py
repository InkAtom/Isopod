from Isopod.Isopod import isopod

def test_match_keypoints():
    isp = isopod(n_oct_layers=10, sigma=2)
    isp.get_image("testimg1.png", "testimg2.png")
    isp.calculate_keypoints()
    isp.match_keypoints(0.5)

    assert len(isp.matches)!=0