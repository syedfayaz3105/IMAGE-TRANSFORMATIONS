# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

### Step2:
Read the input image using cv2.imread() and store it in a variable for further processing.

### Step3:
Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis. 2.Scaling resizes the image by scaling factors. 3.Shearing distorts the image along one axis. 4.Reflection flips the image horizontally or vertically. 5.Rotation rotates the image by a given angle.

### Step4:
Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

### Step5:
Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.
## Program:
```
# Developed By:Farhana H
# Register Number: 2122232320057

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('farhana.jpg')

# Display Original Image
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# i) Image Translation
tx, ty = 100, 50
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, M_translation, (image.shape[1], image.shape[0]))
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.title("Translated Image")
plt.axis('off')

# ii) Image Scaling
fx, fy = 1.5, 1.5
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image")
plt.axis('off')

# iii) Image Shearing
shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1]*2, image.shape[0]*2))
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))
plt.title("Sheared Image")
plt.axis('off')

# iv) Image Reflection
reflected_image = cv2.flip(image, 1)  # 1 for horizontal flip
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB))
plt.title("Reflected Image")
plt.axis('off')

# v) Image Rotation
(height, width) = image.shape[:2]
center = (width // 2, height // 2)
angle = 45  # rotation angle in degrees
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45°)")
plt.axis('off')

plt.show()

# vi) Image Cropping (shown separately to avoid subplot issues)
x, y, w, h = 100, 100, 200, 150
cropped_image = image[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image")
plt.axis('off')
plt.show()

```
## Output:
### i)Image Translation
<img width="404" height="370" alt="Screenshot 2025-09-12 205908" src="https://github.com/user-attachments/assets/f7f52845-4f89-4a09-8e10-a20e83b9de23" />


### ii) Image Scaling
<img width="409" height="381" alt="Screenshot 2025-09-12 205917" src="https://github.com/user-attachments/assets/55d2bc98-812e-4bc0-9db6-3854fa0c02c0" />


### iii)Image shearing

<img width="409" height="381" alt="Screenshot 2025-09-12 205917" src="https://github.com/user-attachments/assets/9ce5c491-189a-4993-9678-92fe9834f9a5" />



### iv)Image Reflection
<img width="413" height="373" alt="Screenshot 2025-09-12 210208" src="https://github.com/user-attachments/assets/a5786f61-7353-4eac-a599-dda473be444a" />




### v)Image Rotation
<img width="443" height="371" alt="Screenshot 2025-09-12 210216" src="https://github.com/user-attachments/assets/eac12417-e4a5-4c72-b954-652ef3512946" />




### vi)Image Cropping
<img width="723" height="522" alt="Screenshot 2025-09-12 210226" src="https://github.com/user-attachments/assets/fd935ca0-8181-4d51-a8c6-52090d113587" />





## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
