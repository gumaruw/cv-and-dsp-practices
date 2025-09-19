import cv2
import numpy as np

# --- CONFIGURATION ---
video_source = "eye_rec0.flv" # video_source = 0 is for webcam

ROI_CONFIG = {
    "eye_rec0.flv": (269, 795, 537, 1416),
    "eye_rec1.flv": (50, 524, 100, 656),
    "eye_rec2.flv": (0, 150, 0, 186)
}

# --- MAIN CODE ---
cap= cv2.VideoCapture(video_source) 
# cap= cv2.VideoCapture("eye_rec0.flv")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- ROI SELECTION ---
    if isinstance(video_source, int) and video_source == 0:
        # if video source is webcam (0): flip the frame horizontally and use full frame
        roi = cv2.flip(frame, 1)
        
    else:
        # if video source is file: use the existing coordinates to crop
        y1, y2, x1, x2 = ROI_CONFIG[video_source]
        roi = frame[y1:y2, x1:x2]

    # print("Original Size:", frame.shape)
    # print("ROI Size:", roi.shape)

    # --- IMAGE PROCESSING ---
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # make the vid gray
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0) # Use a larger kernel for better noise reduction, (1,1) for the unprofessional vids

    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV) 
        
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours


    # --- GAZE & BLINK DETECTION ---
    if contours:
        # If contours are found, the eye is open
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) # Sort by area
        
        # Get the largest contour, which should be the pupil
        cnt = contours[0]
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Calculate the center of the pupil's bounding box
        pupil_center_x = x + w / 2

        # Define boundaries for gaze direction based on the ROI's width
        # The middle 20% of the screen is considered "Center"
        left_boundary = cols * 0.40
        right_boundary = cols * 0.60

        # Determine direction based on the pupil's center position
        if pupil_center_x < left_boundary:
            direction = "Looking Left"
        elif pupil_center_x > right_boundary:
            direction = "Looking Right"
        else:
            direction = "Looking Center"

        # --- DRAWING (only when eye is open) ---
        # Draw rectangle around pupil
        # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3) # draw all contours
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Draw cross lines at the pupil center
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)

    else:
        # If no contours are found, assume the eye is closed
        direction = "Blinking"

    # --- DISPLAY STATUS TEXT ---
    # Put the determined direction text on the ROI frame
    # Using red color for better visibility
    cv2.putText(roi, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2)

    # --- DISPLAY ---
    cv2.imshow("Thresholded ROI", threshold)
    # cv2.imshow("Gray ROI", gray_roi)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()