import cv2
import glob
import numpy as np

def load_Imgs(basePath, filenameList, resize=False, resizeRes=(400,400), inclFilenames = False):
    '''Loads images from filename list
    
    @parameters
    basePath - path to folder containing images.
    filenameList - list of filenames.
    resize = whether to resize the images.
    resizeRes - new resolution of images (leave empty if resize == False).
    includeFilenames - whether to include filename designations in final array.
    
    @returns: list of either just image data or image data and filenames'''
    
    retArr = []
    for flName in filenameList:
        appd = cv2.cvtColor(cv2.imread(f"{basePath}\\{flName}"), cv2.COLOR_BGR2RGB)
        if resize: 
            appd = cv2.resize(appd, resizeRes)
        if inclFilenames:
            appd = (appd, f"{basePath}\\{flName}")
        retArr.append(appd)
    return retArr

def load_folder(folderPath, fileExts):
    '''Load Folder's worth of images.
    
    @parameters:
    folderPath - Path to folder
    fileExts - Extensions of the files you want to load(.jpg, .png etc.). Seperate extensions with | if multiple are needed(.jpg|.png etc.).
    
    @returns: list of images'''
    return [cv2.imread(fName) for fName in glob.glob(f"{folderPath}.{fileExts}")]

def resize_bulk(imageArr, size):
    '''Resizes all images in first parameter to size of second parameter.
    Returns list of resized images'''
    return [cv2.resize(img, size) for img in imageArr]

def clahe(img, thresh):
    '''Applies CLAHE contrast enhancement to image.
    
    @parameters:
    img - image to enhance contrast of.
    thresh - the threshold to start enhancement.
    
    @returns: image with contrast enhanced.'''
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_split = list(cv2.split(lab))

    clahe = cv2.createCLAHE(thresh)
    lab_split[0] = clahe.apply(lab_split[0])

    lab = cv2.merge(lab_split)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb