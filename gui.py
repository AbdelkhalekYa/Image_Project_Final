import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from algorithims import ImageProcessingClass
from tkinter import messagebox
import numpy as np
import cv2
from scipy.signal import find_peaks

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing GUI")
        self.root.geometry("1366x768")  # Adjusted window size for side-by-side images

        # Left frame for buttons and operations
        self.left_frame = tk.Frame(self.root, width=250, bg='darkcyan', padx=20, pady=20)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Buttons for operations
        self.load_button = tk.Button(self.left_frame, text="Load Image", command=self.load_image, width=20, height=1, bg='lavender',  font=("Arial", 12))
        self.load_button.pack(pady=10)

        self.gray_scaling_button = tk.Button(self.left_frame, text="Gray Scaling", command=self.Gray_scaling, width=20, height=1, bg='lavender', font=("Arial", 12))
        self.gray_scaling_button.pack(pady=10)

        self.threshold_button = tk.Button(self.left_frame, text="Threshold", command=self.apply_thresholding, width=20, height=1, bg='lavender',  font=("Arial", 12))
        self.threshold_button.pack(pady=10)

        self.halftone_simple_button = tk.Button(self.left_frame, text="Halftone (Threshold)", command=self.apply_halftone_simple, width=20, height=1, bg='lavender',  font=("Arial", 12))
        self.halftone_simple_button.pack(pady=10)

        self.halftone_button = tk.Button(self.left_frame, text="Halftone", command=self.apply_halftone, width=20, height=1, bg='lavender',  font=("Arial", 12))
        self.halftone_button.pack(pady=10)

        self.histogram_button = tk.Button(self.left_frame, text="Histogram Equalization", command=self.apply_histogram, width=20, height=1, bg='lavender',  font=("Arial", 12))
        self.histogram_button.pack(pady=10)


        # Dropdown for Simple Edge Detection
        tk.Label(self.left_frame, text="Simple Edge Detection", bg='darkcyan', fg='lavender' , font=("Arial", 12)).pack(pady=5)
        self.simple_edge = ttk.Combobox(self.left_frame, values=["Sobel operator", "Prewitt operator", "Kirsch compass"], state="readonly", width=18, font=("Arial", 12))
        self.simple_edge.pack(pady=5)
        self.simple_edge.bind("<<ComboboxSelected>>", self.apply_simple_edge)

        # Dropdown for Advanced Edge Detection
        tk.Label(self.left_frame, text="Advanced Edge Detection", bg='darkcyan', fg='lavender', font=("Arial", 12)).pack(pady=5)
        self.advan_edge = ttk.Combobox(self.left_frame, values=["Homogeneity operator", "Difference operator", "Difference of Gaussian", "Contrast based", "Variance", "Range"], state="readonly", width=18, font=("Arial", 12))
        self.advan_edge.pack(pady=5)
        self.advan_edge.bind("<<ComboboxSelected>>", self.apply_advan_edge)

        # Dropdown for Spatial Frequency
        tk.Label(self.left_frame, text="Spatial Frequency", bg='darkcyan', fg='lavender' ,font=("Arial", 12)).pack(pady=5)
        self.spatial_frequency = ttk.Combobox(self.left_frame, values=["High Pass", "Low Pass", "Median"], state="readonly", width=18, font=("Arial", 12))
        self.spatial_frequency.pack(pady=5)
        self.spatial_frequency.bind("<<ComboboxSelected>>", self.apply_spatial_frequency)

        # Dropdown for Image Operations
        tk.Label(self.left_frame, text="Image Operations", bg='darkcyan', fg='lavender' ,font=("Arial", 12)).pack(pady=5)
        self.image_operations = ttk.Combobox(self.left_frame, values=["Adding", "Subtracting", "Inverting"], state="readonly", width=18, font=("Arial", 12))
        self.image_operations.pack(pady=5)
        self.image_operations.bind("<<ComboboxSelected>>", self.apply_image_operation)

        # Dropdown for Histogram-based Segmentation
        tk.Label(self.left_frame, text="Hist Based Segmentation", bg='darkcyan',fg='lavender' , font=("Arial", 12)).pack(pady=5)
        self.hist_seg = ttk.Combobox(self.left_frame, values=["Manual", "Histogram Peak", "Histogram Valley", "Adaptive Histogram"], state="readonly", width=18, font=("Arial", 12))
        self.hist_seg.pack(pady=5)
        self.hist_seg.bind("<<ComboboxSelected>>", self.apply_hist_seg)

        # Right frame for displaying images
        self.right_frame = tk.Frame(self.root, bg='white', padx=20, pady=20)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Two labels for side-by-side image display
        self.original_image_label = tk.Label(self.right_frame, bg='white', text="Original Image")
        self.original_image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.processed_image_label = tk.Label(self.right_frame, bg='white', text="Processed Image")
        self.processed_image_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Make the grid cells expand to ensure proper centering
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Initialize image variables
        self.image = None
        self.processed_image = None
        
        # Initialize processor
        self.processor = ImageProcessingClass(None)
       
       ###############################################################################



        ###############################################################################
        
        #image loading

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
        if file_path:
            self.image = Image.open(file_path)
            # Update the processor with the loaded image
            self.processor = ImageProcessingClass(self.image)
            self.display_images(self.image, None)

    def display_images(self, original, processed):
        # Resize and display the original image
        if original:
            original_resized = original.resize((500, 500))
            original_tk = ImageTk.PhotoImage(original_resized)
            self.original_image_label.config(image=original_tk)
            self.original_image_label.image = original_tk

        # Resize and display the processed image
        if processed:
            # If the self.processed_image is a tuple, take the first image (or adjust as needed)
            if isinstance(processed, tuple):
                processed = processed[0]  # Get the first image from the tuple
            
            processed_resized = processed.resize((500, 500))
            processed_tk = ImageTk.PhotoImage(processed_resized)
            self.processed_image_label.config(image=processed_tk)
            self.processed_image_label.image = processed_tk
        
        ###############################################################################

        #gray scaling

    def Gray_scaling(self, event=None):
        self.processed_image = None
        self.processed_image = self.processor.grayscale()
        self.display_images(self.image, self.processed_image)

        ###############################################################################

        
        ###############################################################################
        
        #manual segmentation

    def apply_manual_segmentation(self):
        self.processed_image = None

        if self.image is not None:
            # Convert the image to a numpy array in grayscale
            img_array = np.array(self.processor.grayscale())  # Convert to grayscale

            # Set threshold values (you can adjust these as needed)
            low_threshold, high_threshold = 180, 200
            
            # Perform manual segmentation
            segmented_image_array = self.manual_segmentation(img_array, low_threshold, high_threshold)

            # Convert the numpy array back to a PIL Image
            segmented_image = Image.fromarray(segmented_image_array)

            # Display the original and processed images
            self.display_images(self.image, segmented_image)
        
        ###############################################################################
    
    
    
    
        ###############################################################################
        
        #manual segmentation

    def manual_segmentation(self, image, low_threshold, high_threshold, value=255):
        """
        Perform manual image segmentation by thresholding.
        
        :param image: Input image as a numpy array
        :param low_threshold: Lower threshold value
        :param high_threshold: Higher threshold value
        :param value: Value to use for pixels within the threshold range
        :return: Segmented image as a numpy array
        """

        # Create a zeros array with the same shape as the input image
        segmented_image = np.zeros_like(image)
        
        # Create a mask for pixels within the threshold range
        mask = (image >= low_threshold) & (image <= high_threshold)
        
        # Set the pixels within the mask to the specified value
        segmented_image[mask] = value
        
        return segmented_image
        
        ###############################################################################
        

        ###############################################################################
        
        #valley segmentation
    
    def apply_valley_segmentation(self):

        self.processed_image = None
        if self.image is not None:
            # Convert the image to a numpy array in grayscale
            img_array = np.array(self.processor.grayscale())  # Convert to grayscale

            # Perform valley segmentation
            segmented_image_array = self.valley_segmentation(img_array)

            # Convert the numpy array back to a PIL Image
            segmented_image = Image.fromarray(segmented_image_array)

            # Display the original and processed images
            self.display_images(self.image, segmented_image)

    def valley_segmentation(self, image):
        """
        Perform valley-based image segmentation.
        
        :param image: Input image as a numpy array
        :return: Segmented image as a numpy array
        """
        # Compute histogram of the image
        histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=[0, 256])
        
        # Find valleys by inverting the histogram 
        valleys, _ = find_peaks(-histogram, height=0)  # Invert histogram to find valleys
        
        # Determine threshold based on valleys
        if len(valleys) > 1:
            # Use the average of the two lowest valleys as the threshold
            threshold = (bin_edges[valleys[-1]] + bin_edges[valleys[-2]]) / 2
        else:
            # Default threshold if not enough valleys are found
            threshold = 128
        
        # Create binary segmentation
        segmented_image = (image >= threshold) * 255
        
        return segmented_image.astype(np.uint8)
        
        ###############################################################################
        
        
        ###############################################################################
        
        #half tone

    def apply_halftone(self, event=None):
        self.processed_image = None
        self.processed_image = self.processor.halftoning()
        self.display_images(self.image, self.processed_image)
        
        ###############################################################################
        
        
        ###############################################################################
        
        #histogram

    def apply_histogram(self, event=None):
        self.processed_image = None
        self.processed_image = self.processor.histogram_equalization()
        self.display_images(self.image, self.processed_image)
        
        ###############################################################################


        ###############################################################################
        
        #gray scaling

    def Gray_scaling(self, event=None):
        self.processed_image = None
        self.processed_image = self.processor.grayscale()
        self.display_images(self.image, self.processed_image)

        ###############################################################################
        
        
        ###############################################################################
        
        #image operation

    def apply_image_operation(self, event=None):
        if not self.image:
            return
        operation = self.image_operations.get()

        if operation == "Adding":
            self.processed_image = self.processor.add()
        elif operation == "Subtracting":
            self.processed_image = self.processor.subtract()
        elif operation == "Inverting":
            self.processed_image = self.processor.invert()
        else:
            return

        self.display_images(self.image, self.processed_image)

        # Reset the processor with the original image
        self.processor = ImageProcessingClass(self.image)
        
        ###############################################################################


        ###############################################################################

        #advanced edge

    def apply_advan_edge(self, event=None):
        self.processed_image = None
        
        if not self.image:
            return
        operation = self.advan_edge.get()

        if operation == "Homogeneity operator":
            self.processed_image = self.processor.homeginty_operator()
        elif operation == "Difference operator":
            self.processed_image = self.processor.differnce_operator()
        elif operation == "Difference of Gaussian":
            self.processed_image = self.processor.differnece_of_Gaussians()
        elif operation == "Contrast based":
            self.processed_image = self.processor.contrast_based_edge_detection()
        elif operation == "Variance":
            self.processed_image = self.processor.varience_operator()
        elif operation == "Range":
            self.processed_image = self.processor.range_operator()
        else:
            return

        # Ensure self.processed_image is an image object before displaying
        if isinstance(self.processed_image, tuple):
            self.processed_image = self.processed_image[0]  # Adjust index if necessary
        self.display_images(self.image, self.processed_image)
        self.processor = ImageProcessingClass(self.image)

        ###############################################################################


        ###############################################################################       
        
        #filtering

    def apply_spatial_frequency(self, event=None):
        self.processed_image = None
        if not self.image:
            return
        operation = self.spatial_frequency.get()

        if operation == "High Pass":
            self.processed_image = self.processor.high_pass_filter()
        elif operation == "Low Pass":
            self.processed_image = self.processor.low_pass_filter()
        elif operation == "Median":
            self.processed_image = self.processor.median_filter()
        else:
            return

        self.display_images(self.image, self.processed_image)
        self.processor = ImageProcessingClass(self.image)
        
       ###############################################################################


       ###############################################################################
       
        #simple dege

    def apply_simple_edge(self, event=None):
        self.processed_image = None
        if not self.image:
            return
        operation = self.simple_edge.get()

        if operation == "Sobel operator":
            self.processed_image = self.processor.sobel_operator()
        elif operation == "Prewitt operator":
            self.processed_image = self.processor.prewitt_operator()
        elif operation == "Kirsch compass":
            self.processed_image = self.processor.kirsch_compass_operator()
        else:
            return

        self.display_images(self.image, self.processed_image)

        # Reset the processor with the original image
        self.processor = ImageProcessingClass(self.image)

        ###############################################################################


        ###############################################################################
        
        #threshold
        
    def apply_thresholding(self):
        self.processed_image = None
        if not self.image:
            messagebox.showwarning("No Image", "Please load an image before applying thresholding.")
            return
        
        # Convert to a numpy array for processing
        img_array = np.array(self.image)

        # Calculate the average pixel value
        avg_pixel_value = np.mean(img_array)

        # Show the average pixel value in a pop-up message box
        messagebox.showinfo("Average Pixel Value", f"The average pixel value is: {avg_pixel_value:.2f}")
        
        ###############################################################################



        ###############################################################################
        
        #histogram segmentation
    
    def apply_hist_seg(self, event=None):
        if not self.image:
            return
        operation = self.hist_seg.get()

        if operation == "Manual":
             self.processed_image = self.apply_manual_segmentation()
        elif operation == "Histogram Peak":
             self.processed_image = self.apply_peak_segmentation()
        elif operation == "Histogram Valley":
             self.processed_image = self.apply_valley_segmentation()
        elif operation == "Adaptive Histogram":
            self.processed_image = self.apply_adaptive_segmentation()
        else:
            return

        self.display_images(self.image, self.processed_image)
        self.processor = ImageProcessingClass(self.image)

        ###############################################################################



        ###############################################################################

        #simple halftone

    def apply_halftone_simple(self):
        self.processed_image = None
        if not self.image:
            messagebox.showwarning("No Image", "Please load an image before applying the halftone effect.")
            return

        # Process the image
        threshold = 128  # Default threshold; adjust as needed
        self.processed_image = self.processor.halftoning_simple(threshold)

        # Display the processed image
        self.display_images(self.image, self.processed_image)

        ###############################################################################


        ###############################################################################

        #peak segmentation
    
    def apply_peak_segmentation(self):
        self.processed_image = None
        if self.image is not None:
            # Convert the image to a numpy array in grayscale
            img_array = np.array(self.processor.grayscale())  # Convert to grayscale

            # Compute histogram of the image
            histogram, bin_edges = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
            
            # Find peaks in the histogram
            peaks, _ = find_peaks(histogram, height=0)
            
            # Determine threshold based on peaks
            if len(peaks) > 0:
                # Use the highest peak's corresponding intensity as threshold
                threshold = bin_edges[peaks[-1]]
            else:
                # Default threshold if no peaks found
                threshold = 128
            
            # Create binary segmentation
            segmented_image_array = (img_array >= threshold) * 255
            
            # Convert the numpy array back to a PIL Image
            segmented_image = Image.fromarray(segmented_image_array.astype(np.uint8))

            # Display the original and processed images
            self.display_images(self.image, segmented_image)

            ###############################################################################


            ###############################################################################

        #adaptive segmentation

    def apply_adaptive_segmentation(self):
        self.processed_image = None
        if self.image is not None:
            # Convert the image to a numpy array in grayscale
            img_array = np.array(self.processor.grayscale())  # Convert to grayscale

            # Perform adaptive thresholding using OpenCV
            adaptive_thresh = cv2.adaptiveThreshold(
                img_array, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11,  # Block size
                2    # Constant subtracted from the mean
            )
            
            # Convert the numpy array back to a PIL Image
            segmented_image = Image.fromarray(adaptive_thresh)

            # Display the original and processed images
            self.display_images(self.image, segmented_image)

            ###############################################################################


def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()