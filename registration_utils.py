import cv2
import numpy as np

def nparraytoimg(arr):
    arr.clip(min=0)
    arr = arr/arr.max()
    #arr = 255 * arr
    return arr.astype(np.uint8)

def estimateAffineWarpMatrix(image, reference):
    # Read the images to be aligned

    # Convert images to grayscale
    im1_gray = reference
    im2_gray = image


    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None, 1)

    return warp_matrix# Show final results

def warp_image(image, matrix):
    sz = image.shape
    return cv2.warpAffine(image, matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

