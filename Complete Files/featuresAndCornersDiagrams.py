'''Generates and plots SIFT, ORB, Shi tomasi, kMeans clustering and Harris corner detection.

Change path in line 11 to change input image'''


import cv2 as cv2
import matplotlib.pyplot as plt
import featureAndCornerDetectors as fcd
import imageManipulationUtils as imu

img = cv2.imread("Media/Input/IMG_1293.JPG")

nImg = img

sift = fcd.sift(nImg, 2000, sized=True)
orb = fcd.orb(nImg)
shi = fcd.shiTomasi(nImg)
kmeans = fcd.kMeans(nImg, iterations=1)
harris = fcd.harrisCorners(nImg, [0, 255, 0])

features = [["SIFT", sift], ["ORB", orb], ["Shi-Tomasi", shi], ["kMeans Clustering", kmeans],
             ["Harris", harris]]

cImg = imu.clahe(img, 5)

cSift = fcd.sift(cImg, 2000, sized=True)
cOrb = fcd.orb(cImg)
cShi = fcd.shiTomasi(cImg)
cKmeans = fcd.kMeans(cImg, iterations=1)
cHarris = fcd.harrisCorners(cImg, [0, 255, 0])

cFeatures = [["Clahe + \nSIFT", cSift], ["Clahe + \nORB", cOrb], ["Clahe + \nShi-Tomasi", cShi], ["Clahe + \nkMeans Clustering", cKmeans],
             ["Clahe + \nHarris", cHarris]]

# ---------------------------------------------------------
# PLOTTING

plt.xticks([])
plt.yticks([])

cFig, cAxes = plt.subplots(nrows=1, ncols=5)

cFig.tight_layout(pad=0.5)

for i, v in enumerate(cFeatures):
    cAxes[i].set_title(v[0]) 
    cAxes[i].imshow(v[1])

fig, axes = plt.subplots(nrows=1, ncols=5)

fig.tight_layout(pad=0.5)

for i, v in enumerate(features):
    axes[i].set_title(v[0]) 
    axes[i].imshow(v[1])

plt.show()

# ---------------------------------------------------------

cv2.imwrite("Media/Output/Features and corners/sift.jpg", sift)
cv2.imwrite("Media/Output/Features and corners/orb.jpg", orb)
cv2.imwrite("Media/Output/Features and corners/shiTomasi.jpg", shi)
cv2.imwrite("Media/Output/Features and corners/harris.jpg", harris)
cv2.imwrite("Media/Output/Features and corners/kMeans Clustering.jpg", kmeans)

cv2.imwrite("Media/Output/Features and corners/clahe_sift.jpg", cSift)
cv2.imwrite("Media/Output/Features and corners/clahe_orb.jpg", cOrb)
cv2.imwrite("Media/Output/Features and corners/clahe_shiTomasi.jpg", cShi)
cv2.imwrite("Media/Output/Features and corners/clahe_harris.jpg", cHarris)
cv2.imwrite("Media/Output/Features and corners/clahe_kMeans Clustering.jpg", cKmeans)