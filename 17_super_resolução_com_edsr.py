import cv2

# Read image
path_image = 'fig_01.JPG'
img = cv2.imread(path_image)

# Model EDSR
sr = cv2.dnn_superres.DnnSuperResImpl_create() 
path = "EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr",4)
result = sr.upsample(img)
 
# Resized image
resized = cv2.resize(img, dsize=None, fx=4, fy=4)

# Save the result image
cv2.imwrite("result_01.jpg", result)

# Save the resized image
cv2.imwrite("resized_01.jpg", resized)