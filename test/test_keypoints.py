import sys
sys.path.append('/home/brainless/Master/COSMOS/Isopod/Isopod')
import matplotlib.pyplot as plt


from Isopod import isopod

def test_compare_keypoints():
    isp = isopod(n_oct_layers=10, sigma=2.4)
    isp.get_image("testimg1.png", "testimg2.png")
    isp.calculate_keypoints()


    print(isp.keypoints)

    plt.imshow(isp.grayscale_images[1])
    for kp in isp.keypoints[1]:
        plt.scatter(kp.pt[0],kp.pt[1],marker='x', c='r')
    plt.show()

test_compare_keypoints()

