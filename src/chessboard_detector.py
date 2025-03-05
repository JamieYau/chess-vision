import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

WIDTH, HEIGHT = 1200, 1200 # Size of the output image
ROWS, COLS = 8, 8 # Size of the chessboard
SQUARE_SIZE = HEIGHT // ROWS # Size of each square

def preprocess_image(image):
    """
    Preprocess the image to extract the chessboard
    
    :param image: The image to preprocess
    :return: The preprocessed image
    """
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_image,(5,5),0)
    ret, otsu_binary = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_image = cv2.Canny(otsu_binary, 20, 255)
    kernel = np.ones((7, 7), np.uint8)
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    # Hough Lines (find straight lines)
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)
    # Create an image that contains only black pixels
    black_image = np.zeros_like(dilation_image)
    # Draw only lines that are output of HoughLinesP function to the "black_image"
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw only lines to the "black_image"
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.dilate(black_image, kernel, iterations=1)

    return black_image

def find_contours(black_image):
    """
    Find the contours of the chessboard
    
    :param black_image: The image to find the contours
    :return: The contours of the chessboard
    """
    board_contours, hierarchy = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    square_centers=list()

    # draw filtered rectangles to image for visualization
    board_squares = np.zeros_like(black_image)

    for contour in board_contours:
        if 4000 < cv2.contourArea(contour) < 20000:
            # Approximate the contour to a simpler shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure the approximated contour has 4 points (quadrilateral)
            if len(approx) == 4:
                pts = [pt[0] for pt in approx]  # Extract coordinates

                # Define the points
                pt1, pt2, pt3, pt4 = pts

                x, y, w, h = cv2.boundingRect(contour)
                center_x=(x+(x+w))/2
                center_y=(y+(y+h))/2

                square_centers.append([center_x,center_y,pt2,pt1,pt3,pt4])

                # Draw the lines between the points
                cv2.line(board_squares, pt1, pt2, (255, 255, 0), 7)
                cv2.line(board_squares, pt1, pt3, (255, 255, 0), 7)
                cv2.line(board_squares, pt2, pt4, (255, 255, 0), 7)
                cv2.line(board_squares, pt3, pt4, (255, 255, 0), 7)

    return board_squares

def find_biggest_contour(board_squares):
    """
    Find the biggest contour of the chessboard
    
    :param board_squares: The image to find the biggest contour
    :return: The biggest contour of the chessboard
    """
    # Apply dilation to the valid_squares_image
    kernel = np.ones((7, 7), np.uint8)
    dilated_valid_squares_image = cv2.dilate(board_squares, kernel, iterations=1)

    # Find contours of dilated_valid_squares_image
    contours, _ = cv2.findContours(dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if contours:
        # take biggest contour 
        largest_contour = max(contours, key=cv2.contourArea)

        # create black image
        biggest_area_image = np.zeros_like(dilated_valid_squares_image)

        # Initialize variables to store extreme points
        top_left, top_right, bottom_left, bottom_right = None, None, None, None

        # Loop through the contour to find extreme points
        for point in largest_contour[:, 0]:
            x, y = point

            if top_left is None or (x + y < top_left[0] + top_left[1]):
                top_left = (x, y)

            if top_right is None or (x - y > top_right[0] - top_right[1]):
                top_right = (x, y)

            if bottom_left is None or (x - y < bottom_left[0] - bottom_left[1]):
                bottom_left = (x, y)

            if bottom_right is None or (x + y > bottom_right[0] + bottom_right[1]):
                bottom_right = (x, y)

        return top_left, top_right, bottom_right, bottom_left

def warp_image(image, top_left, top_right, bottom_right, bottom_left):
    """
    Warp the image to the chessboard
    
    :param image: The image to warp
    :param top_left: The top-left corner of the chessboard
    :param top_right: The top-right corner of the chessboard
    :param bottom_right: The bottom-right corner of the chessboard
    :param bottom_left: The bottom-left corner of the chessboard
    :return: The warped image
    """
    # Prepare the source points
    src_points = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype='float32')
    # Prepare the destination points
    dst_points = np.array([
        [0, 0],  # Top-left
        [WIDTH, 0],  # Top-right
        [WIDTH, HEIGHT],  # Bottom-right
        [0, HEIGHT]  # Bottom-left
    ], dtype='float32')


    # Calculate the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the perspective transform to the original image
    warped_image = cv2.warpPerspective(image, perspective_matrix, (WIDTH, HEIGHT))

    return warped_image, src_points, perspective_matrix

def draw_chessboard(image):
    """
    Draw the chessboard on the image
    
    :param image: The image to draw the chessboard
    :param perspective_matrix: The perspective matrix of the chessboard
    :param src_points: The source points of the chessboard
    """
    for row in range(ROWS):
        for col in range(COLS):
            top_left = (col * SQUARE_SIZE, row * SQUARE_SIZE)
            bottom_right = ((col+1) * SQUARE_SIZE, (row+1) * SQUARE_SIZE)
            
            # Draw the square
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    return image

def extract_squares(image, perspective_matrix, src_points):
    """
    Extract the squares from the chessboard
    
    :param image: The image to extract the squares
    :param perspective_matrix: The perspective matrix of the chessboard
    :param src_points: The source points of the chessboard
    """
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Compute the inverse perspective transformation matrix
    M_inv = cv2.invert(perspective_matrix)[1]  # Get the inverse of the perspective matrix

    # List to store squares' data in the correct order (bottom-left first)
    squares_data_warped = []

    for i in range(ROWS - 1, -1, -1):  # Start from bottom row and move up
        for j in range(COLS):  # Left to right order
            # Define the 4 corners of each square
            top_left = (j * SQUARE_SIZE, i * SQUARE_SIZE)
            top_right = ((j + 1) * SQUARE_SIZE, i * SQUARE_SIZE)
            bottom_left = (j * SQUARE_SIZE, (i + 1) * SQUARE_SIZE)
            bottom_right = ((j + 1) * SQUARE_SIZE, (i + 1) * SQUARE_SIZE)

            # Calculate center of the square
            x_center = (top_left[0] + bottom_right[0]) // 2
            y_center = (top_left[1] + bottom_right[1]) // 2

            # Append to list in the correct order
            squares_data_warped.append([
                (x_center, y_center),
                bottom_right,
                top_right,
                top_left,
                bottom_left
            ])

    # Convert to numpy array for transformation
    squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)

    # Transform all points back to the original image
    squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)

    # Reshape back to list format
    squares_data_original = squares_data_original_np.reshape(-1, 5, 2)

    for square in squares_data_original:
        x_center, y_center = tuple(map(int, square[0]))
        bottom_right = tuple(map(int, square[1]))
        top_right = tuple(map(int, square[2]))
        top_left = tuple(map(int, square[3]))
        bottom_left = tuple(map(int, square[4]))

        # Draw necessary lines only (to form grid)
        cv2.line(rgb_image, top_left, top_right, (0, 255, 0), 6)  # Top line
        cv2.line(rgb_image, top_left, bottom_left, (0, 255, 0), 6)  # Left line

        # Draw bottom and right lines only for last row/column
        if j == COLS - 1:
            cv2.line(rgb_image, top_right, bottom_right, (0, 255, 0), 8)  # Right line
        if i == 0:
            cv2.line(rgb_image, bottom_left, bottom_right, (0, 255, 0), 8)  # Bottom line

    cv2.circle(rgb_image, (int(src_points[0][0]),int(src_points[0][1])), 15, (255, 255, 255), -1)   
    cv2.circle(rgb_image,  (int(src_points[1][0]),int(src_points[1][1])), 15, (255, 255, 255), -1)   
    cv2.circle(rgb_image,  (int(src_points[2][0]),int(src_points[2][1])), 15, (255, 255,255), -1)   
    cv2.circle(rgb_image,  (int(src_points[3][0]),int(src_points[3][1])), 15, (255, 255, 255), -1)   

    return rgb_image
    
def main(image_path, output_path, visualize=False):
    """
    Main function to run the chessboard detector
    
    :param image_path: The path to the image
    :param output_path: The path to the output image
    :param visualize: Whether to visualize the chessboard
    """
    image = cv2.imread(image_path)
    dilation_image = preprocess_image(image)
    contours = find_contours(dilation_image)
    
    top_left, top_right, bottom_right, bottom_left = find_biggest_contour(contours)
    warped_image, src_points, perspective_matrix = warp_image(image, top_left, top_right, bottom_right, bottom_left)
    chessboard_image = draw_chessboard(warped_image)
    overlay_image = extract_squares(image, perspective_matrix, src_points)

    # if output_path not exists, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    success = cv2.imwrite(os.path.join(output_path, "chessboard.png"), overlay_image)
    if not success:
        print(f"Failed to save image to: {output_path}")

    if visualize:
        plt.figure(figsize=(9, 7))
        plt.imshow(image)
        plt.title("Original Image")
        plt.show()
        
        plt.figure(figsize=(9, 7))
        plt.imshow(warped_image)
        plt.title("Warped Image")
        plt.show()
        
        plt.figure(figsize=(9, 7))
        plt.imshow(chessboard_image)
        plt.title("Warped Chessboard Divided into 64 Squares")
        plt.show()
        
        plt.figure(figsize=(9, 7))
        plt.imshow(overlay_image)
        plt.title("Original Image with Chessboard")
        plt.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("output_path", type=str, default="output", const="output", nargs='?', help="Path to the output directory")
    parser.add_argument("-v", "--visualize",  type=bool, default=False, const=True, nargs='?', help="Visualize the chessboard")
    args = parser.parse_args()
    main(args.image_path, args.output_path, args.visualize)
