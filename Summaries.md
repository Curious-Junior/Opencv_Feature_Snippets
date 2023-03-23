# Summaries of feature functions

## Contours

### Practical:

* Apply the threshold function in opencv to convert image to binary for better results.
* Then use the findContours function in opencv on the binary image.
* Now, plot the contours using the drawContours function in opencv.

### Theoretical:

* Convert the image to binary by applying a threshold as it yields better results.
* Apply a contour genereation algorithm to detect the contours.  
  Opencv uses this [algorithm](https://www.sciencedirect.com/science/article/abs/pii/0734189X85900167).

## Blobs

### Practical:

- Create and configure the SimpleBlobDetector from opencv using a SimpleBlobDetector_Params object.
- Now use the SimpleBlobDetector.detect() function on a grayscale version of your image.
- Then draw the returned keypoints on your image using the drawKeypoints function in opencv.

### Theoretical:

#### The most common approach to blob detection is the Laplacian of Gaussian.

- Apply a gaussian blur to input image by convolving over it with a gaussian kernel.

* A gaussian kernel can be generated with the following equation:  
  $g(x,y,t) = \frac{1}{2\pi t}\ e^-\frac{x^2+y^2}{2t}\$

* This convolution is performed with a specific scale _t_. This results in a scale-space representation of the image.
* After that, a Laplacian operator is applied to the images to bring out the blobs.
* The simplest Laplacian operator in this case is $\Delta L = L_{xx} + L_{yy}$  
  This provides a strong positive response for dark blobs and a strong negative response for bright blobs.  
  The drawback to this method is that it only detects blobs of size $r^2 = 2t$ or $r^2 = dt$ where _d_ = number of dimensions.
* To get multi-scale blobs, the following equation is used: $\Delta^2_{norm} L(x,y;t) = t(L_{xx} + L_{yy})$

## Ridges:

### Practical:

- Create a RidgeDetectionFilter object in opencv.
- Use the RidgeDetectionFilter.getRidgeFilteredImage() function on your image.

### Theoretical:

**The mathematical definition of a ridge is very complex, please see [wikipedia](https://en.wikipedia.org/wiki/Ridge_detection#Differential_geometric_definition_of_ridges_and_valleys_at_a_fixed_scale_in_a_two-dimensional_image) for a adequate explanation.**  
The computation of a ridge however is relatively simpler:

- The intensity of a ridge is defined by the following equation:  
  ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/8966ab3703612cf38dd49239d338e9829e78ac84)
- Where _t_ is the scale-space representation of the image.
- The main principal curvature of the ridge is defined by the following equation:  
  ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3cafac8e1bc8b41dcaa0a480023f3a5015c615e6)

## Harris Corners:

### Practical:

- Use the cornerHarris function in opencv on a grayscale version of your image.
- Dilate the image using opencv's dilate function to make corners visible on image.
- Use _img[dst>0.01*harris.max()]=[0,0,255]_ to apply a threshold to your corners, resulting in only the good ones remaining.

### Theoretical:

- The first step is to convert the input image to grayscale.
- Secondly, two derivatives are calculated, one with respect to _x_, $I_x(x,y)$ and one with respect to _y_, $I_y(x,y)$.
- Then, with derivatives $I_x(x,y)$ and $I_y(x,y)$ a tensor _M_ is constructed.
- Now, the Harris response calculation is applied to _M_.

* A commonly used Harris response calculation is:  
  $R = \lambda _1 \lambda _2 - k(\lambda _1 + \lambda _2)^2 = det(M) - k{tr}(M)^2$  
  Where _k_ is an empirically detemined constant; $k \in [0.04, 0.06]$

## Shi-Tomasi Corners:

### Practical:

- Use the goodFeaturesToTrack in opencv on a grayscale version of your image.
- Go though each of the corners in the returned list with a for loop and plot every single one on the image with opencv's circle function.

### Theoretical:

- The first step is to convert the input image to grayscale.
- Secondly, two derivatives are calculated, one with respect to _x_, $I_x(x,y)$ and one with respect to _y_, $I_y(x,y)$.
- Then, with derivatives $I_x(x,y)$ and $I_y(x,y)$ a tensor _M_ is constructed.
- Now, the Shi-Tomasi response calculation is applied to _M_.  
  $R = min(\lambda _1, \lambda _2)$
  Where _k_ is an empirically detemined constant; $k \in [0.04, 0.06]$

_The Shi-Tomasi response calculation is an altercation of the Harris response calculation.  
All the steps are identical to Harris corners except the calculation._

## Blurring image:

### Practical:

- Use the GaussianBlur function in opencv on your image.

### Theoretical:

- Convolve over the image with a Gaussian Kernel.  
  A Gaussian kernel can be generated with the following equation:
  $g(x,y,t) = \frac{1}{2\pi t}\ e^-\frac{x^2+y^2}{2t}\$

## Generating a Scale Space:

### Practical and Theoretical:

- Apply the GaussianBlur function in opencv on your image multiple times with increasing size of blur kernel.

#### These resulting images are called a octave.

- Scale down the image(by a factor of your choice, usually half).
- Apply GaussianBlur on scaled-down image.

#### This is another octave.

#### _Scale down the image once again and repeat._

## Generating a DoG(Difference of gaussian)

### Practical and Theoretical:

- Take the first and second image in a octave and subtract the second one from first using the subtract function in opencv.
- Now do the same with the second and third image and so on.

This is to be repeated for each octave.
