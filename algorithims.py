import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks

class ImageProcessingClass:

    def __init__(self, image):
        self.image = image


#################################################################################################################################

# Grayscale , Halftone , Histogram

    def grayscale(self):
        """Convert the image to grayscale."""
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        img_array = np.array(self.image)
        # Compute grayscale values using the luminance formula
        grayscale_array = (
            0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        ).astype(np.uint8)

        self.image = Image.fromarray(grayscale_array)
        return self.image
    
#######################################################
    
    def halftoning(self, threshold=128):
        if self.image.mode != 'L':
            self.image = self.grayscale()
        
        img_array = np.array(self.image, dtype=np.float32)

        height, width = img_array.shape
        for i in range(height):
            for j in range(width):
                old_pixel = img_array[i, j]
                new_pixel = 255 if old_pixel >= threshold else 0
                img_array[i, j] = new_pixel
                error = old_pixel - new_pixel

                if j + 1 < width:
                    img_array[i, j + 1] += error * 7 / 16
                if i + 1 < height and j > 0:
                    img_array[i + 1, j - 1] += error * 3 / 16
                if i + 1 < height:
                    img_array[i + 1, j] += error * 5 / 16
                if i + 1 < height and j + 1 < width:
                    img_array[i + 1, j + 1] += error * 1 / 16

        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))

#######################################################


    def histogram_equalization(self):
        if self.image.mode != 'L':
            self.image = self.grayscale()       
        img_array = np.array(self.image)
        flat = img_array.flatten()

        hist, bins = np.histogram(flat, bins=256, range=[0, 256], density=True)
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])

        equalized_img_array = np.interp(flat, bins[:-1], cdf_normalized).reshape(img_array.shape)
        return Image.fromarray(equalized_img_array.astype(np.uint8))



#################################################################################################################################

    # Simple Edge Detection 

    def sobel_operator(self):
        """Apply Sobel operator manually."""
        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))
        height, width = img_array.shape

        # Sobel kernels
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Pad the image
        padded_image = np.pad(img_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        # Initialize the output array
        sobel_image = np.zeros((height, width), dtype=float)

        # Perform convolution
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                region = padded_image[i - 1:i + 2, j - 1:j + 2]
                gx = np.sum(region * Gx)
                gy = np.sum(region * Gy)
                sobel_image[i - 1, j - 1] = np.sqrt(gx**2 + gy**2)

        # Normalize to 0-255
        sobel_image = ((sobel_image - sobel_image.min()) / (sobel_image.max() - sobel_image.min()) * 255).astype(np.uint8)
        self.image = Image.fromarray(sobel_image)
        return self.image

#######################################################

    def prewitt_operator(self):
        """Apply Prewitt operator manually."""
        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))
        height, width = img_array.shape

        # Prewitt kernels
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # Pad the image
        padded_image = np.pad(img_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        # Initialize the output array
        prewitt_image = np.zeros((height, width), dtype=float)

        # Perform convolution
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                region = padded_image[i - 1:i + 2, j - 1:j + 2]
                gx = np.sum(region * Gx)
                gy = np.sum(region * Gy)
                prewitt_image[i - 1, j - 1] = np.sqrt(gx**2 + gy**2)

        # Normalize to 0-255
        prewitt_image = ((prewitt_image - prewitt_image.min()) / (prewitt_image.max() - prewitt_image.min()) * 255).astype(np.uint8)
        self.image = Image.fromarray(prewitt_image)
        return self.image

#######################################################

    def kirsch_compass_operator(self):
        """Apply Kirsch Compass operator manually."""
        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))
        height, width = img_array.shape

        # Kirsch masks
        kirsch_masks = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
        ]

        # Pad the image
        padded_image = np.pad(img_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        # Initialize the output array
        kirsch_image = np.zeros((height, width), dtype=float)

        # Perform convolution
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                region = padded_image[i - 1:i + 2, j - 1:j + 2]
                responses = [np.sum(region * mask) for mask in kirsch_masks]
                kirsch_image[i - 1, j - 1] = max(responses)

        # Normalize to 0-255
        kirsch_image = ((kirsch_image - kirsch_image.min()) / (kirsch_image.max() - kirsch_image.min()) * 255).astype(np.uint8)
        self.image = Image.fromarray(kirsch_image)
        return self.image
    
#################################################################################################################################



# filtering high , low , medain  


    def high_pass_filter(self):

        mask = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        if self.image.mode != 'L':
            self.image = self.grayscale()

        img_array = np.array(self.image, dtype=np.float32)
        result = cv2.filter2D(img_array, -1, mask)
        result = np.clip(result, 0, 255)
        return Image.fromarray(result.astype(np.uint8))

#######################################################


    def low_pass_filter(self):

        mask = np.array([
            [0, 1/6, 0],
            [1/6, 2/6, 1/6],
            [0, 1/6, 0]
        ], dtype=np.float32)

        if self.image.mode != 'L':
            self.image = self.grayscale()

        img_array = np.array(self.image, dtype=np.float32)
        result = cv2.filter2D(img_array, -1, mask)
        result = np.clip(result, 0, 255)
        return Image.fromarray(result.astype(np.uint8))

#######################################################


    def median_filter(self):

        if self.image.mode != 'L':
            self.image = self.grayscale()

        img_array = np.array(self.image, dtype=np.uint8)
        filtered_image = cv2.medianBlur(img_array, 3)
        return Image.fromarray(filtered_image)
    
    

#################################################################################################################################

# Image Operations
    
    def add(self):
        """Add the image to itself."""
        self.image = self.grayscale()
        image_array = np.array(self.image)

        # Check if the image is grayscale or RGB
        if len(image_array.shape) == 2:  # Grayscale image
            height, width = image_array.shape
            channels = 1
            image_array = image_array[:, :, None]  # Add a dummy channel for uniform processing
        else:  # RGB image
            height, width, channels = image_array.shape

        # Initialize the result image
        added_image = np.zeros_like(image_array, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    added_image[i, j, k] = min(image_array[i, j, k] + image_array[i, j, k], 255)

        # If the original image was grayscale, remove the dummy channel
        if channels == 1:
            added_image = added_image[:, :, 0]

        self.image = Image.fromarray(added_image)
        return self.image
    
#######################################################
    
    def subtract(self):
        """Subtract the image from itself."""
        self.image = self.grayscale()
        image_array = np.array(self.image)

        # Check if the image is grayscale or RGB
        if len(image_array.shape) == 2:  # Grayscale image
            height, width = image_array.shape
            channels = 1
            image_array = image_array[:, :, None]  # Add a dummy channel for uniform processing
        else:  # RGB image
            height, width, channels = image_array.shape

        # Initialize the result image
        subtracted_image = np.zeros_like(image_array, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    subtracted_image[i, j, k] = max(image_array[i, j, k] - image_array[i, j, k], 0)

        # If the original image was grayscale, remove the dummy channel
        if channels == 1:
            subtracted_image = subtracted_image[:, :, 0]

        self.image = Image.fromarray(subtracted_image)
        return self.image

#######################################################

    def invert(self):
        """Invert the image."""
        self.image = self.grayscale()
        image_array = np.array(self.image)

        # Check if the image is grayscale or RGB
        if len(image_array.shape) == 2:  # Grayscale image
            height, width = image_array.shape
            channels = 1
            image_array = image_array[:, :, None]  # Add a dummy channel for uniform processing
        else:  # RGB image
            height, width, channels = image_array.shape

        # Initialize the result image
        inverted_image = np.zeros_like(image_array, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    inverted_image[i, j, k] = 255 - image_array[i, j, k]

        # If the original image was grayscale, remove the dummy channel
        if channels == 1:
            inverted_image = inverted_image[:, :, 0]

        self.image = Image.fromarray(inverted_image)
        return self.image


#################################################################################################################################

# Advanced Edge Detection

    def homeginty_operator(self, threshold=5):
        """Apply Homeginty operator for edge detection"""

        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))  # Convert to grayscale
        height, width = img_array.shape
        homeginty_image = np.zeros_like(img_array)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center_pixel = img_array[i, j]
                differences = [
                    abs(center_pixel - img_array[i - 1, j - 1]),
                    abs(center_pixel - img_array[i - 1, j]),
                    abs(center_pixel - img_array[i - 1, j + 1]),
                    abs(center_pixel - img_array[i, j - 1]),
                    abs(center_pixel - img_array[i, j + 1]),
                    abs(center_pixel - img_array[i + 1, j - 1]),
                    abs(center_pixel - img_array[i + 1, j]),
                    abs(center_pixel - img_array[i + 1, j + 1]),
                ]
                homogeneity_value = max(differences)
                homeginty_image[i, j] = homogeneity_value
                homeginty_image[i, j] = np.where(homeginty_image[i, j] >= threshold, homeginty_image[i, j], 0)

        return Image.fromarray(homeginty_image)

#######################################################

    def differnce_operator(self, threshold=10):
        """Apply Difference operator for edge detection"""

        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))  # Convert to grayscale
        height, width = img_array.shape
        differnce_image = np.zeros_like(img_array)

        for i in range(1, height - 1):
            for j in range(1, width - 2):
                dif1 = abs(img_array[i - 1, j - 1] - img_array[i + 1, j + 1])
                dif2 = abs(img_array[i - 1, j - 1] - img_array[i + 1, j - 1])
                dif3 = abs(img_array[i - 1, j - 1] - img_array[i, j + 1])
                dif4 = abs(img_array[i - 1, j - 1] - img_array[i + 1, j])
                max_diff = max(dif1, dif2, dif3, dif4)
                differnce_image[i, j] = max_diff
                differnce_image[i, j] = np.where(max_diff >= threshold, max_diff, 0)

        return Image.fromarray(differnce_image)
    
#######################################################

    def contrast_based_edge_detection(self):
        """Apply Contrast-Based Edge Detection from scratch"""

        self.image = self.grayscale()

        # Convert image to grayscale and numpy array
        img_array = np.array(self.image.convert('L'))  # Convert to grayscale

        # Define edge detection and smoothing masks
        edge_mask = np.array([[-1, 0, -1],
                            [0, 4, 0],
                            [-1, 0, -1]])
        smoothing_mask = np.ones((3, 3)) / 9

        # Get image dimensions
        height, width = img_array.shape
        pad_h, pad_w = edge_mask.shape[0] // 2, edge_mask.shape[1] // 2

        # Pad the image
        padded_image = np.pad(img_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Initialize output arrays
        edge_output = np.zeros_like(img_array, dtype=float)
        average_output = np.zeros_like(img_array, dtype=float)

        # Perform convolution for edge detection and smoothing
        for i in range(height):
            for j in range(width):
                # Extract the region of interest
                region = padded_image[i:i + edge_mask.shape[0], j:j + edge_mask.shape[1]]
                # Compute edge detection and smoothing responses
                edge_output[i, j] = np.sum(region * edge_mask)
                average_output[i, j] = np.sum(region * smoothing_mask)

        # Avoid division by zero in contrast calculation
        average_output += 1e-10

        # Compute contrast edges
        contrast_edges = edge_output / average_output

        # Normalize results to 0-255 range for display
        min_val, max_val = np.min(contrast_edges), np.max(contrast_edges)
        contrast_edges = ((contrast_edges - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        min_val, max_val = np.min(edge_output), np.max(edge_output)
        edge_output = ((edge_output - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        min_val, max_val = np.min(average_output), np.max(average_output)
        average_output = ((average_output - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Convert arrays back to PIL images
        return Image.fromarray(contrast_edges), Image.fromarray(edge_output), Image.fromarray(average_output)


#######################################################


    def differnece_of_Gaussians(self):
        """Apply Difference of Gaussians (DoG) for edge detection"""

        self.image = self.grayscale()

        mask1 = np.array([
            [0,0,-1,-1,-1,0,0],
            [0,-2,-3,-3,-3,-2,0],
            [-1,-3,5,5,5,-3,-1],
            [-1,-3,5,16,5,-3,-1],
            [-1,-3,5,5,5,-3,-1],
            [0,-2,-3,-3,-3,-2,0],
            [0,0,-1,-1,-1,0,0]
        ],dtype=np.float32)

        mask2 = np.array([
            [0,0,0,-1,-1,-1,0,0,0],
            [0,-2,-3,-3,-3,-3,-3,-2,0],
            [0,-3,-2,-1,-1,-1,-2,-3,0],
            [-1,-3,-1,9,9,9,-1,-3,-1],
            [-1,-3,-1,9,19,9,-1,-3,-1],
            [-1,-3,-1,9,9,9,-1,-3,-1],
            [0,-3,-2,-1,-1,-1,-2,-3,0],
            [0,-2,-3,-3,-3,-3,-3,-2,0],
            [0,0,0,-1,-1,-1,0,0,0]
        ],dtype=np.float32)
        img_array = np.array(self.image.convert('L'))  # Convert to grayscale
        blured1 = cv2.filter2D(img_array, -1, mask1)
        blured2 = cv2.filter2D(img_array, -1, mask2)

        dog = blured1 - blured2
        return Image.fromarray(dog), Image.fromarray(blured1), Image.fromarray(blured2)
    
#######################################################

    def varience_operator(self):
        """Apply Variance operator for edge detection"""

        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))  # Convert to grayscale
        height, width = img_array.shape
        output = np.zeros_like(img_array)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = img_array[i - 1:i + 2, j - 1:j + 2]
                mean = np.mean(neighborhood)
                variance = np.sum((neighborhood - mean) ** 2) / 9
                output[i, j] = variance

        return Image.fromarray(output)

#######################################################

    def range_operator(self):
        """Apply Range operator for edge detection"""

        self.image = self.grayscale()
        img_array = np.array(self.image.convert('L'))  # Convert to grayscale
        height, width = img_array.shape
        output = np.zeros_like(img_array)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = img_array[i - 1:i + 2, j - 1:j + 2]
                range_value = np.max(neighborhood) - np.min(neighborhood)
                output[i, j] = range_value

        return Image.fromarray(output)

#######################################################

    def halftoning_simple(self, threshold=128):
        # Convert image to grayscale
        grayscale_image = self.image.convert("L")
        # Apply a threshold (halftone effect)
        halftone_image = grayscale_image.point(lambda p: 255 if p > threshold else 0)
        return halftone_image
    
#################################################################################################################################
