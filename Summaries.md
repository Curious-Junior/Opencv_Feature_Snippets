# Summaries of featur functions

## Contours

### Practical:

- Apply a threshold to input image and then use the findCountours function in opencv.

### Theoretical:

## Blobs

### Practical:

- Create and configure the SimpleBlobDetector from opencv using a SimpleBlobDetector_Params object.
- Now use the SimpleBlobDetector.detect() function on a grayscale version of your image.
- Then draw the returned keypoints on your image using the drawKeypoints function in opencv.

### Theoretical:

## Ridges:

### Practical:

- Create a RidgeDetectionFilter object in opencv.
- Use the RidgeDetectionFilter.getRidgeFilteredImage() function on your image.

### Theoretical:

## Harris Corners:

### Practical:

- Use the cornerHarris function in opencv on a grayscale version of your image.
- Dilate the image using opencv's dilate function to make corners visible on image.
- Use img[dst>0.01*harris.max()]=[0,0,255] to apply a threshold to your corners, resulting in only the good ones remaining.

### Theoretical

## Shi-Tomasi Corners:

### Practical:

- Use the goodFeaturesToTrack in opencv on a grayscale version of your image.
- Go though each of the corners in the returned list with a for loop and plot every single one on the image with opencv's cirlce function.

### Theoretical:

## Blurring image:

### Practical:

- Use the GaussianBlur function in opencv on your image.

### Theoretical:

## Generating a Scale Space:

### Practical:

- Apply the GaussianBlur function in opencv on your image multiple times with inreasing size of blur kernel.

###### These resulting images are called a octave.

- Scale down the image(by a factor of your choice, usually half).
- Apply GaussianBlur on scaled-down image.

###### This is another octave.

##### Scale down the image once again and repeat.

### Theoretical:

## Generating a DoG(Difference of gaussian)

### Practical:

- Take the first and second image in a octave and subtract the second one from first using the subtract function in opencv.
- Now do the same with the second and third image and so on.

This is to be repeated for each octave.

### Theoretical:
