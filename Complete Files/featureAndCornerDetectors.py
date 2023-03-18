import cv2
import numpy as np

try:
    import xfeatures2d as xf2
except ImportError:
    print("Features and Corner Detectors: xfeatures2d not found, SURF will be unavailable.")



def sift(img, noOfKeys = 2000, keyColor = (0, 0, 255), retDesc = False, sized=False):
    '''Detects features in an image using SIFT(Scale-Invariant Feature Transfrom).
    
    @parameters:
    img - image to search for features.
    noOfKeys - number of features to detect(keypoint is synonymous to features). Enter -1 for all keys detected
    keyColor - color of the plotted features, format is (Blue, Green, Red), values between 0 and 255.
    retDesc - if to return the descriptor object along with the keypoints.
    
    @returns: Image with features plotted, if retDesc is true returns plotted image and descriptor object tuple.'''

    image = np.array(img) # Make a copy of the image
    res = image # Create variable that will hold the resulting image
    sift = cv2.SIFT_create(2000, 4) # Create SIFT detector object

    key, des = sift.detectAndCompute(image, None) # Detect SIFT features

    key = key[:noOfKeys] # Limit number of features based on input

    # Draw keypoints and draw rich keypoints(keypoints with size) if the user demands it
    cv2.drawKeypoints(image, key, res, color=keyColor, flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if sized else None) 
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # Convert color channel order forn BLUE-RED-GREED(BGR) to the standard RED-GRENN-BLUE(RGB)

    # Conditional raw keypoint return for user convenience
    if retDesc:
        return (cv2.cvtColor(res, cv2.COLOR_BGR2RGB), key, des)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

# This is a patented alogrithm and requires the opencv-contrib git repostitory
# to be built from source with the ENABLE_NONFREE parameter set to ON
# Use this at your own risk
def surf(img, noOfKeys = 2000, keyColor = (0, 0, 255), retDesc = False, sized=False):
    '''Detects features in an image using SURF(Speed Up Robust Features).
    
    @parameters:
    img - image to search for features.
    noOfKeys - number of features to detect(keypoint is synonymous to features). Enter -1 for all keys detected
    keyColor - color of the plotted features, format is (Blue, Green, Red), values between 0 and 255.
    retDesc - if to return the descriptor object along with the keypoints.
    
    @returns: Image with features plotted, if retDesc is true returns plotted image and descriptor object tuple.'''
    
    if (not xf2): return -1 # Return early if the opencv-contrib repo and hence the xfeatures2d(xf2) are not properly installed 
    
    image = np.array(img) # Make a copy of the image
    res = image # Create variable that will hold the resulting image
    surf = xf2.SURF_create(nOctaves=4) # Create SURF detector object
    
    key, des = surf.detectAndCompute(image, None) # Detect SURF features
    
    key = key[:noOfKeys] # Limit number of features based on input
   
    # Draw keypoints and draw rich keypoints(keypoints with size) if the user demands it
    cv2.drawKeypoints(image, key, res, color=keyColor, flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if sized else None) 
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # Convert color channel order forn BLUE-RED-GREED(BGR) to the standard RED-GRENN-BLUE(RGB)

    if retDesc:
        return (res, key)
    return res

def orb(img, noOfKeys = 2000, keyColor = (0, 0, 255), retDesc = False):
    '''Detects features in an image using ORB(Oriented FAST and rotated BRIEF).
    
    @parameters:
    img - image to search for features.
    noOfKeys - number of features to detect(keypoint is synonymous to features). Enter -1 for all keys detected
    keyColor - color of the plotted features, format is (Blue, Green, Red), values between 0 and 255.
    retDesc - if to return the descriptor object along with the keypoints.
    
    @returns: Image with features plotted, if retDesc is true returns plotted image and descriptor object tuple.'''

    image = np.array(img) # Make a copy of the Image
    res = image # Create variable that will hold the resulting image
    orb = cv2.ORB_create(noOfKeys) # Create SURF detector object

    key, des = orb.detectAndCompute(image, None) # Detect ORB features

    key = key[:noOfKeys] # Limit number of features based on input
   
    # Draw keypoints and draw rich keypoints(keypoints with size) if the user demands it
    cv2.drawKeypoints(image, key, res, color=keyColor, flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if sized else None) 
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # Convert color channel order forn BLUE-RED-GREED(BGR) to the standard RED-GRENN-BLUE(RGB)

    if retDesc:
        return (res, des)
    return res

def match_features(img1, key1, desc1, img2, key2, desc2, richKeys = False, filterDistance = 10, matchColor = (255, 0, 0), pointColor = (0, 0, 255)):
    '''Matches the selected "good" features of an image.
    
    @parameters: 
    img1, key1, desc1 - the first image along with its keypoints and descriptor.
    img2, key2, desc2 - the second image along with its keypoints and descriptor.
    richKeys - Whether to plot size(intensity) of the keypoints.
    filterDistance - The cut-off for detecting "good" keypoints, all keypoints with distance lower than this will be discarded.
    matchColor - Color of the match lines.
    pointColor - Color of the matched as well as unmatched keypoints.
    
    @returns: An image showing the similarities between the two input images.'''

    matcher = cv2.BFMatcher()
    matches = matcher.match(desc1, desc2)

    goodMatches = [m for m in matches if m.distance <= filterDistance]

    out = cv2.drawMatches(img1, key1, img2, key2, goodMatches, None, matchColor, pointColor, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if richKeys else None)
    return (out)


def detect_Ridges(image):
    '''Detects the ridges in an image(not the same as edges of an image).
    
    @parameters: 
    image - input image
    
    @returns: Image with ridges highlighted'''

    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create() # Create tehe ridge detection object
    ridges = ridge_filter.getRidgeFilteredImage(image) # Filter out the ridges
    return ridges


def kMeans(image, K=10, resizeFactor=2.0, type=cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, iterations=10, accuracy=1.0, centers=cv2.KMEANS_RANDOM_CENTERS):
    '''Applies kMeans clustering to the inputted image.
    
    @parameters:
    image - image to apply kMeans to.
    resizeFactor - Factor to scale image up or down with.
    type - criteria for the algorithm [DONT CHANGE UNLESS YOU KNOW WHAT IT IS].
    iterations - number of times the algorithm will be applied
    accuracy - accuracy parameter for algorithm [DONT CHANGE UNLESS YOU KNOW WHAT IT IS].
    centers - method to acquire center of neighbourhood [DONT CHANGE UNLESS YOU KNOW WHAT IT IS].
    
    @returns: Image with kmeans clustering applied.'''

    height, width, _ = image.shape
    resizeShape = (int(width * resizeFactor), int(height * resizeFactor))
    image = cv2.resize(image, resizeShape)

    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (type, iterations, accuracy)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, centers)
    if not ret: return -1
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    res2 = cv2.resize(res2, (width, height))
    return res2

def shiTomasi(img, noOfCorners = 2000, color = (0, 0, 255), retCorners=False):
    '''Detects corners in an image using shi-Tomasi corner detection.
    
    @parameters:
    img - image to search for corners.
    noOfCorners - number of corners to detect.
    color - color of the plotted corners, format is (Blue, Green, Red), values between 0 and 255.
    retCorners - wether to return the raw corner data or not.
    
    @returns: Image with corners plotted, if retCorners is true returns plotted image along with raw corner data in the form of a tuple.'''

    image = np.array(img) # Make a copy of the image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Apply grayscale colors to image
    corners = cv2.goodFeaturesToTrack(gray, noOfCorners, 0.1, 10) # Detect shi-Tomasi corners

    corners = np.int0(corners) # Change data type of corners to int0 --- MIGHT BE REDUNDANT ---

    # Go through the corners and plot each one as a circle
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 5, color, -1)

    # Conditional raw data return for user convenience
    if retCorners:
        return image, corners
    return image

def harrisCorners(img, color = [0, 0, 255]):
    '''Detects corners in an image using the harris corner detection method,
    
    @parameters:
    img - Image to detect the corners of.
    color - color of the plotted corners, format is (Blue, Green, Red), values between 0 and 255.

    @returns: Image with corners plotted.'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Turn image into grayscale colors
    harris = cv2.cornerHarris(np.float32(gray), 2,3,0.04) # Detect the corners with Harris method

    dst = cv2.dilate(harris,None) # Dilate the results to make them visible
    img[dst>0.01*harris.max()]=[0,0,255] # Apply a optimal-value threshold to only use good corners

    return img

def contours(img, color, retCont=False):
    '''Detects contours in an image.

    @parameters:
    img - image to search for contours.
    color - color of the plotted contours, format is (Blue, Green, Red), values between 0 and 255.
    retCont - whether to return the raw contours data or not.
    
    @returns: Image with contours plotted, if retCont is true returns plotted image along with raw contour data in the form of a tuple.'''

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Apply grayscale color to image
    ret, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV) # Apply binary threshold to image
    if not ret: return -1 # If an error occurs, return early to avoid crashes

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Detect contours

    cv2.drawContours(img, contours, -1, color, 3) # Draw contours onto image

    # Optional raw data return for user convenience
    if retCont:
        return (img, contours)
    return img

def detect_Blobs(image, retKeys = False):
    '''Detects blobs in an image.

    @parameters:
    img - image to search for blobs.
    color - color of the plotted blobs, format is (Blue, Green, Red), values between 0 and 255.
    retKeys - whether to return the raw keypoint data or not.
    
    @returns: Image with blobs plotted, if retKeys is true returns plotted image along with raw blobs data in the form of a tuple.'''
    imgRGB = image # Make image copy
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Apply grayscale color to image
    params = cv2.SimpleBlobDetector_Params() # Create a blob detector object
    
    # Set parameters
    params.filterByColor = True 
    params.filterByArea = True
    params.minArea = 0
    params.maxArea = 100

    params.filterByConvexity = True
    params.minConvexity = 0.2

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Feed parameters into blob detector object
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image) # Detect blobs

    blank = np.zeros((1,1)) # Blank vector
    blobs = cv2.drawKeypoints(imgRGB, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw detected blobs
    if retKeys:
        return (blobs, keypoints)
    return blobs

