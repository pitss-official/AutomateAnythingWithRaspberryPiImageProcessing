# Real-time Human Face, Eyes and Noise Detection

# Import Computer Vision package - cv2
import cv2

# Import Numerical Python package - numpy as np
import numpy as np

# Load human face cascade file using cv2.CascadeClassifier built-in function
# cv2.CascadeClassifier([filename]) 
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Load human eyes cascade file using cv2.CascadeClassifier built-in function
# cv2.CascadeClassifier([filename]) 
eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load human noise cascade file using cv2.CascadeClassifier built-in function
# cv2.CascadeClassifier([filename]) 
noise_detect = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Check if human face cascade file is loaded
if face_detect.empty():
	raise IOError('Unable to haarcascade_frontalface_alt.xml file')

# Check if human eyes cascade file is loaded
if eyes_detect.empty():
	raise IOError('Unable to load haarcascade_eye.xml file')
	
# Check if human noise cascade file is loaded	
if noise_detect.empty():
	raise IOError('Unable to load haarcascade_mcs_nose.xml file')
	
# Initializing video capturing object
capture = cv2.VideoCapture(0)
# One camera will be connected by passing 0 OR -1
# Second camera can be selected by passing 2


# Initialize While Loop and execute until Esc key is pressed
while True:
	# Start capturing frames
	ret, capturing = capture.read()
	
	# Resize the frame using cv2.resize built-in function
    # cv2.resize(capturing, output image size, x scale, y scale, interpolation)
	resize_frame = cv2.resize(capturing, None, fx=0.5, fy=0.5, 
            interpolation=cv2.INTER_AREA)
   
    # Convert RGB to gray using cv2.COLOR_BGR2GRAY built-in function
	# BGR (bytes are reversed)
	# cv2.cvtColor: Converts image from one color space to another
	gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)

    # Detect objects(faces) of different sizes using cv2.CascadeClassifier.detectMultiScale
    # cv2.CascadeClassifier.detectMultiScale(gray, scaleFactor, minNeighbors)
   
    # scaleFactor: Specifies the image size to be reduced
    # Faces closer to the camera appear bigger than those faces in the back.
    
    # minNeighbors: Specifies the number of neighbors each rectangle should have to retain it
    # Higher value results in less detections but with higher quality
    
	face_detection = face_detect.detectMultiScale(gray, 1.3, 5)
	# Rectangles are drawn around the face image using cv2.rectangle built-in function
	# cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
	for (x,y,w,h) in face_detection:
		cv2.rectangle(resize_frame, (x,y), (x+w,y+h), (0,0,255), 10)    
		# Find the Region Of Interest (ROI) in color image and grayscale image
	gray_roi = gray[y:y+h, x:x+w]
	color_roi = resize_frame[y:y+h, x:x+w]
       
    # Apply eye detector on the grayscale Region Of Interest (ROI)
	eye_detector = eyes_detect.detectMultiScale(gray_roi)
        
    # Rectangles are drawn around the color eyes image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
		
	for (eye_x, eye_y, eye_w, eye_h) in eye_detector:
		cv2.rectangle(color_roi,(eye_x,eye_y),(eye_x + eye_w, eye_y + eye_h),(255,0,0),5)
			           
	# Apply nose detector in the grayscale ROI
	nose_detector = noise_detect.detectMultiScale(gray_roi, 1.3, 5)

    # Rectangles are drawn around noise using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
	for (nose_x, nose_y, nose_w, nose_h) in nose_detector:
		cv2.rectangle(color_roi, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0,255,0), 5)

    # Display the eyes detected using imshow built-in function
	cv2.imshow("Real-time Detection", resize_frame)

    # Check if the user has pressed Esc key
	c = cv2.waitKey(1)
	if c == 27:
		break

# Close the capturing device
capture.release()

# Close all windows
cv2.destroyAllWindows()
