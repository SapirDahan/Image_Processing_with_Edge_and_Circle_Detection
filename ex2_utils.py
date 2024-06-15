import numpy as np
from matplotlib import pyplot as plt
import cv2


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    kernel1 = np.flip(kernel1)  # Flip the kernel
    output_len = len(inSignal) + len(kernel1) - 1
    result = np.zeros(output_len)

    # Padding the input signal
    padded_signal = np.pad(inSignal, (len(kernel1) - 1, len(kernel1) - 1), 'constant')

    # Convolution operation
    for i in range(output_len):
        result[i] = np.sum(padded_signal[i:i + len(kernel1)] * kernel1)

    return result

def conv1Demo():
    inSignal = np.array([1, 2, 3, 4, 5])
    kernel1 = np.array([1, 0, -1])

    convolved_signal = conv1D(inSignal, kernel1)

    print("Input Signal:", inSignal)
    print("Kernel:", kernel1)
    print("Convolved Signal:", convolved_signal)

    # Plotting the signals
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.stem(inSignal, use_line_collection=True)
    plt.title('Input Signal')

    plt.subplot(3, 1, 2)
    plt.stem(kernel1, use_line_collection=True)
    plt.title('Kernel')

    plt.subplot(3, 1, 3)
    plt.stem(convolved_signal, use_line_collection=True)
    plt.title('Convolved Signal')

    plt.tight_layout()
    plt.show()


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    kernel2 = np.flipud(np.fliplr(kernel2))  # Flip the kernel
    output_shape = (inImage.shape[0] + kernel2.shape[0] - 1, inImage.shape[1] + kernel2.shape[1] - 1)
    result = np.zeros(output_shape)

    # Padding the input image
    padded_image = np.pad(inImage,
                          ((kernel2.shape[0] - 1, kernel2.shape[0] - 1), (kernel2.shape[1] - 1, kernel2.shape[1] - 1)),
                          'constant')

    # Convolution operation
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(padded_image[i:i + kernel2.shape[0], j:j + kernel2.shape[1]] * kernel2)

    return result

def conv2Demo():
    inImage = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    kernel2 = np.array([[1, 0],
                        [0, -1]])

    convolved_image = conv2D(inImage, kernel2)

    print("Input Image:\n", inImage)
    print("Kernel:\n", kernel2)
    print("Convolved Image:\n", convolved_image)

    # Plotting the images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(inImage, cmap='gray', interpolation='none')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(kernel2, cmap='gray', interpolation='none')
    plt.title('Kernel')

    plt.subplot(1, 3, 3)
    plt.imshow(convolved_image, cmap='gray', interpolation='none')
    plt.title('Convolved Image')

    plt.tight_layout()
    plt.show()


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude, x_der, y_der)
    """
    # Define the kernels for x and y derivatives
    kernel_x = np.array([1, 0, -1]).reshape((1, 3))
    kernel_y = np.array([1, 0, -1]).reshape((3, 1))

    # Convolve the image with the kernels to get derivatives
    x_der = conv2D(inImage, kernel_x)
    y_der = conv2D(inImage, kernel_y)

    # Crop the derivatives to ensure they have the same shape
    min_shape = (min(x_der.shape[0], y_der.shape[0]), min(x_der.shape[1], y_der.shape[1]))
    x_der = x_der[:min_shape[0], :min_shape[1]]
    y_der = y_der[:min_shape[0], :min_shape[1]]

    # Compute the magnitude of the gradient
    magnitude = np.sqrt(x_der ** 2 + y_der ** 2)

    # Compute the direction of the gradient
    directions = np.arctan2(y_der, x_der)

    return directions, magnitude, x_der, y_der


def derivDemo():
    # Create a simple grayscale image
    inImage = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

    # Compute the derivatives, magnitude, and direction
    directions, magnitude, x_der, y_der = convDerivative(inImage)

    # Print the results
    print("Input Image:\n", inImage)
    print("X Derivative:\n", x_der)
    print("Y Derivative:\n", y_der)
    print("Magnitude:\n", magnitude)
    print("Directions:\n", directions)

    # Plot the results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(inImage, cmap='gray', interpolation='none')
    plt.title('Input Image')

    plt.subplot(2, 3, 2)
    plt.imshow(x_der, cmap='gray', interpolation='none')
    plt.title('X Derivative')

    plt.subplot(2, 3, 3)
    plt.imshow(y_der, cmap='gray', interpolation='none')
    plt.title('Y Derivative')

    plt.subplot(2, 3, 4)
    plt.imshow(magnitude, cmap='gray', interpolation='none')
    plt.title('Magnitude')

    plt.subplot(2, 3, 5)
    plt.imshow(directions, cmap='gray', interpolation='none')
    plt.title('Directions')

    plt.tight_layout()
    plt.show()


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """

    def create_gaussian_kernel(size: int) -> np.ndarray:
        """Create a 2D Gaussian kernel using binomial coefficients"""
        pascal = np.array([1])
        for _ in range(size - 1):
            pascal = np.convolve(pascal, [1, 1])
        kernel_1d = pascal / pascal.sum()
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / kernel_2d.sum()

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size)

    # Pad the image
    padded_image = cv2.copyMakeBorder(in_image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                      cv2.BORDER_REPLICATE)

    # Perform convolution
    blurred_image = conv2D(padded_image, kernel)

    return blurred_image


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel_2d = np.outer(kernel, kernel)

    # Blur the image using OpenCV's filter2D function
    blurred_image = cv2.filter2D(in_image, -1, kernel_2d, borderType=cv2.BORDER_REPLICATE)

    return blurred_image


def blurDemo(image_path: str, kernel_size: int):
    """
    Demonstrate the blurring of an image using both blurImage1 and blurImage2
    :param image_path: Path to the input image
    :param kernel_size: Kernel size for the Gaussian blur
    """
    # Load the image
    in_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if in_image is None:
        print("Error: Image not found or unable to load.")
        return

    # Apply the custom blur
    blurred_image1 = blurImage1(in_image, kernel_size)

    # Apply the OpenCV blur
    blurred_image2 = blurImage2(in_image, kernel_size)

    # Display the results
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(in_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_image1, cmap='gray')
    plt.title('Blurred Image - Custom Implementation')

    plt.subplot(1, 3, 3)
    plt.imshow(blurred_image2, cmap='gray')
    plt.title('Blurred Image - OpenCV Implementation')

    plt.show()


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    # OpenCV Sobel implementation
    sobelx_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_cv = np.sqrt(sobelx_cv**2 + sobely_cv**2)
    sobel_cv = (sobel_cv > thresh * np.max(sobel_cv)).astype(np.uint8) * 255

    # My implementation
    sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    sobel_x = cv2.filter2D(img_padded, -1, sobelx)
    sobel_y = cv2.filter2D(img_padded, -1, sobely)

    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = (sobel > thresh * np.max(sobel)).astype(np.uint8) * 255

    return sobel_cv, sobel


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    zero_crossing = np.zeros_like(laplacian)

    # Detect zero crossings
    for i in range(1, laplacian.shape[0] - 1):
        for j in range(1, laplacian.shape[1] - 1):
            patch = laplacian[i - 1:i + 2, j - 1:j + 2]
            if np.max(patch) * np.min(patch) < 0:
                zero_crossing[i, j] = 255

    return zero_crossing


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    log = cv2.Laplacian(blurred_img, cv2.CV_64F)
    zero_crossing_log = np.zeros_like(log)

    # Detect zero crossings
    for i in range(1, log.shape[0] - 1):
        for j in range(1, log.shape[1] - 1):
            patch = log[i - 1:i + 2, j - 1:j + 2]
            if np.max(patch) * np.min(patch) < 0:
                zero_crossing_log[i, j] = 255

    return zero_crossing_log


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges using "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    # OpenCV Canny implementation
    canny_cv = cv2.Canny(img, thrs_1, thrs_2)

    # My implementation (placeholder, typically you'd use the OpenCV implementation)
    canny = cv2.Canny(img, thrs_1, thrs_2)  # Placeholder for a custom implementation if needed

    return canny_cv, canny


def edgeDemo(image_path: str, sobel_thresh: float, canny_thrs1: float, canny_thrs2: float):
    """
    Demonstrate the edge detection using various methods
    :param image_path: Path to the input image
    :param sobel_thresh: Threshold for Sobel edge detection
    :param canny_thrs1: Threshold1 for Canny edge detection
    :param canny_thrs2: Threshold2 for Canny edge detection
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or unable to load.")
        return

    # Sobel Edge Detection
    sobel_cv, sobel_my = edgeDetectionSobel(img, sobel_thresh)

    # Zero Crossing Simple Edge Detection
    zero_crossing_simple = edgeDetectionZeroCrossingSimple(img)

    # Zero Crossing LOG Edge Detection
    zero_crossing_log = edgeDetectionZeroCrossingLOG(img)

    # Canny Edge Detection
    canny_cv, canny_my = edgeDetectionCanny(img, canny_thrs1, canny_thrs2)

    # Display the results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 3, 2)
    plt.imshow(sobel_cv, cmap='gray')
    plt.title('Sobel Edge Detection (OpenCV)')

    plt.subplot(2, 3, 3)
    plt.imshow(sobel_my, cmap='gray')
    plt.title('Sobel Edge Detection (Custom)')

    plt.subplot(2, 3, 4)
    plt.imshow(zero_crossing_simple, cmap='gray')
    plt.title('Zero Crossing Simple')

    plt.subplot(2, 3, 5)
    plt.imshow(zero_crossing_log, cmap='gray')
    plt.title('Zero Crossing LOG')

    plt.subplot(2, 3, 6)
    plt.imshow(canny_cv, cmap='gray')
    plt.title('Canny Edge Detection (OpenCV)')

    plt.tight_layout()
    plt.show()


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float, param1: float = 1, param2: float = 40,
                min_dist: float = 100) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :param param1: Higher threshold for Canny edge detector (lower threshold is twice smaller)
    :param param2: Accumulator threshold for the circle centers at the detection stage
    :param min_dist: Minimum distance between the centers of the detected circles
    :return: A list containing the detected circles, [(x, y, radius), (x, y, radius), ...]
    """
    # Apply Gaussian Blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (9, 9), 2)

    # Apply Canny Edge Detector
    edges = cv2.Canny(blurred_img, 100, 200)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=int(min_radius), maxRadius=int(max_radius))

    # Prepare the list of circles
    circle_list = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_list.append((x, y, r))

    return circle_list


def houghDemo(image_path: str, min_radius: float, max_radius: float):
    """
    Demonstrate the Hough Circle detection
    :param image_path: Path to the input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    """


    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)

    if img is None:
        print("Error: Image not found or unable to load.")
        return

    # Detect circles
    circles = houghCircle(img, min_radius, max_radius)
    print("Detected circles:", circles)

    # Draw circles on the image and display
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in circles:
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 4)

    # Display the result
    plt.imshow(output_img)
    plt.title('Hough Circle Detection')
    plt.show()