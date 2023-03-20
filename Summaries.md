# Summaries of feature functions

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

#### The most common approach to blob detection is the Laplacian of Gaussian.

- Apply a gaussian blur to input image by convolving over it with a gaussian kernel.

* An gaussian kernel can be generated with the following equation:  
  $g(x,y,t) = \frac{1}{2\pi t}\ e^-\frac{x^2+y^2}{2t}\$

* This convolution is performed with a specific scale _t_. This results in a scale-space representation of the image.
* After that, a Laplacian operator is applied to the images to bring out the blobs.
* The simplest Laplacian operator in this case is $\Delta L = L_xx + L_yy$  
  This provides a strong postive response for dark blobs and a strong negative response for bright blobs.  
  The drawback to this method is that it only detects blobs of size $r^2 = 2t$ or $r^2 = dt$ where $d = {number of dimensions}$.
* To get multi-scale blobs, the following equation is used: $\Delta ^2 _{norm} L(x,y;t) = t(L_xx + L_yy)$

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
- Go though each of the corners in the returned list with a for loop and plot every single one on the image with opencv's circle function.

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
