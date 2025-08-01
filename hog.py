

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return filter_x, filter_y

def filter_image(im, filter):
    return cv2.filter2D(im, -1, filter)

def get_gradient(im_dx, im_dy):
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    grad_angle = np.arctan2(im_dy, im_dx) % (2 * np.pi)
    return grad_mag, grad_angle

def build_histogram(grad_mag, grad_angle, cell_size):
    num_cells_y, num_cells_x = grad_mag.shape[0] // cell_size, grad_mag.shape[1] // cell_size
    ori_histo = np.zeros((num_cells_y, num_cells_x, 6))
    bin_width = np.pi / 6

    for i in range(num_cells_y):
        for j in range(num_cells_x):
            cell_mag = grad_mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_angle = grad_angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            histogram = np.zeros(6)

            for k in range(cell_mag.shape[0]):
                for l in range(cell_mag.shape[1]):
                    angle = cell_angle[k, l]
                    mag = cell_mag[k, l]
                    bin_idx = int(np.floor(angle / bin_width)) % 6
                    histogram[bin_idx] += mag

            ori_histo[i, j] = histogram

    return ori_histo

def get_block_descriptor(ori_histo, block_size):
    num_cells_y, num_cells_x, num_bins = ori_histo.shape
    num_blocks_y = num_cells_y - block_size + 1
    num_blocks_x = num_cells_x - block_size + 1
    ori_histo_normalized = np.zeros((num_blocks_y, num_blocks_x, block_size**2 * num_bins))

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = ori_histo[i:i+block_size, j:j+block_size].flatten()
            norm_block = block / (np.sqrt(np.sum(block**2)) + 1e-6)
            ori_histo_normalized[i, j] = norm_block

    return ori_histo_normalized

def extract_hog(im, cell_size=8, block_size=2):
    im = im.astype('float32') / 255.0  # Use float32 for precision
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    hog = get_block_descriptor(ori_histo, block_size)
    return hog.ravel()

def sliding_window(image, window_size, step_size):
    (h, w) = image.shape[:2]
    (winH, winW) = window_size
    for y in range(0, h - winH + 1, step_size):
        for x in range(0, w - winW + 1, step_size):
            yield (x, y, image[y:y + winH, x:x + winW])

def detect_similar_regions(target_img, template_features, window_size, step_size=16, similarity_threshold=0.3):
    similar_regions = []
    for (x, y, window) in sliding_window(target_img, window_size, step_size):
        window_resized = cv2.resize(window, (window_size[1], window_size[0]), interpolation=cv2.INTER_AREA)
        window_features = extract_hog(window_resized)

        # Debugging: Print shapes of features
        print(f"Window features shape: {window_features.shape}, Template features shape: {template_features.shape}")

        if window_features.shape[0] == template_features.shape[0]:
            similarity = np.dot(window_features, template_features) / (np.linalg.norm(window_features) * np.linalg.norm(template_features))
            print(f"Similarity: {similarity}")
            if similarity > similarity_threshold:
                similar_regions.append((x, y, window_size[0], window_size[1], similarity))
    return similar_regions

def non_maximum_suppression(bounding_boxes, threshold=0.5):
    if len(bounding_boxes) == 0:
        return []

    boxes = np.array(bounding_boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return boxes[keep]

def visualize_face_detection(I_target, bounding_boxes):
    if len(I_target.shape) == 2:
        I_target = cv2.cvtColor(I_target, cv2.COLOR_GRAY2RGB)
    fimg = I_target.copy()
    for ii in range(bounding_boxes.shape[0]):
        x, y, w, h = bounding_boxes[ii, :4].astype(int)
        fimg = cv2.rectangle(fimg, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if bounding_boxes.shape[1] > 4:
            cv2.putText(fimg, "%.2f" % bounding_boxes[ii, 4], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Face Detection')
    plt.show()

if __name__ == '__main__':
    I_target = cv2.imread('/content/target.png', cv2.IMREAD_GRAYSCALE)
    I_template = cv2.imread('/content/template.png', cv2.IMREAD_GRAYSCALE)

    if I_target is None or I_template is None:
        raise ValueError("Could not load one or both images. Please check the file paths.")

    # Adjust template to match the window size
    window_size = (128, 64)  # Set this based on your expected window dimensions
    template_resized = cv2.resize(I_template, (window_size[1], window_size[0]), interpolation=cv2.INTER_AREA)

    # Extract HOG features from the resized template image
    template_features = extract_hog(template_resized, cell_size=8, block_size=2)

    # Detect similar regions in the target image
    similar_regions = detect_similar_regions(I_target, template_features, window_size, step_size=16, similarity_threshold=0.3)

    # Apply Non-Maximum Suppression to filter out overlapping boxes
    filtered_regions = non_maximum_suppression(similar_regions, threshold=0.5)

    # Visualize the target image with bounding boxes for similar regions
    visualize_face_detection(I_target, np.array(filtered_regions))

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_face_detection(I_target, bounding_boxes):
    # Convert the target image to color if it is grayscale
    if len(I_target.shape) == 2:
        I_target = cv2.cvtColor(I_target, cv2.COLOR_GRAY2BGR)

    fimg = I_target.copy()
    for (x, y, w, h) in bounding_boxes:
        # Draw rectangle around detected face
        fimg = cv2.rectangle(fimg, (x, y), (x + w, y + h), (0, 0, 0), 2)

    plt.imshow(cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis('off')  # Hide axes
    plt.title('Face Detection')
    plt.show()

def face_recognition(I_target):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Preprocess the image (optional)
    I_target = cv2.equalizeHist(I_target)  # Improve contrast in the image

    # Detect faces in the target image with adjusted parameters
    faces = face_cascade.detectMultiScale(
        I_target,
        scaleFactor=1.05,  # Adjust scaleFactor
        minNeighbors=3,    # Adjust minNeighbors
        minSize=(20, 20)   # Adjust minSize
    )

    # Print the number of faces detected and their coordinates for debugging
    print(f"Number of faces detected: {len(faces)}")
    for (x, y, w, h) in faces:
        print(f"Face coordinates: x={x}, y={y}, w={w}, h={h}")

    return faces

if __name__ == '__main__':
    # Load images in grayscale for face recognition
    I_target = cv2.imread('/content/target.png', cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if I_target is None:
        raise ValueError("Could not load the target image. Please check the file path.")

    # Perform face recognition
    bounding_boxes = face_recognition(I_target)

    # Load the color version of the target image for visualization
    I_target_color = cv2.imread('/content/target.png')  # Load the color version for visualization

    # Visualize the target image with bounding boxes
    visualize_face_detection(I_target_color, bounding_boxes)

    # Load the Cameraman image
    im = cv2.imread('/content/Cameraman.tif', 0)  # Load in grayscale

    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap='gray')
    plt.title('Original Cameraman Image')
    plt.axis('off')
    plt.show()

    # Extract and visualize HOG features
    hog = extract_hog(im)