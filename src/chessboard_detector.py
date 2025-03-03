import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """
    Load an image and apply grayscale conversion & Gaussian blur.
    """
    image = cv2.imread(image_path)  # OpenCV loads in BGR format
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return image, gray_image, blurred_image


def detect_edges(blurred_image):
    """
    Apply OTSU thresholding and Canny edge detection.
    """
    # OTSU thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    edges = cv2.Canny(binary_image, 20, 255)

    # Dilation to enhance lines
    kernel = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    return dilated_edges


def detect_lines(dilated_edges):
    """
    Apply Hough Transform to detect straight lines in the image.
    """
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)

    # Create a black image of the same size
    line_image = np.zeros_like(dilated_edges)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dilation to enhance detected lines
    kernel = np.ones((3, 3), np.uint8)
    line_image = cv2.dilate(line_image, kernel, iterations=1)

    return line_image


def plot_results(original, processed, title="Processed Image"):
    """
    Display original and processed images side by side.
    """
    plt.figure(figsize=(9, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap="gray")
    plt.title(title)

    plt.show()


def process_chessboard(image_path):
    """
    Complete pipeline to process a chessboard image.
    """
    original, gray, blurred = preprocess_image(image_path)
    edges = detect_edges(blurred)
    lines = detect_lines(edges)

    plot_results(original, lines, title="Detected Chessboard Lines")


# Example Usage (Can be removed in production)
if __name__ == "__main__":
    process_chessboard("../data/test-images/test-10.jpeg")
