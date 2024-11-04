
import numpy as np
import cv2

def postprocess_image(segmented_img):
    # Convert segmented image to binary for watershed
    # ret1, thresh = cv2.threshold(segmented_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure_bg = cv2.dilate(opening, kernel, iterations=10)

    # Distance transform to identify sure foreground
    dist_transform = cv2.distanceTransform(segmented_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4* dist_transform.max(), 255, 0)#like erosing
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(segmented_img, sure_fg)

    # Marker labeling for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR), markers)

    return markers

def postprocess_prediction(predicted_mask):

    mask_probabilities = predicted_mask[0]

    binary_mask = (mask_probabilities > 0.99).astype(np.uint8)

    # Postprocess and analyze segmented image
    markers = postprocess_image(binary_mask)


    ppr_markers = markers.copy()
    for y in range(ppr_markers.shape[0]):
        for x in range(ppr_markers.shape[1]):
            if binary_mask[y, x] == 0:
                ppr_markers[y, x] = 0
    return ppr_markers