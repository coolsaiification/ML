import cv2

img = cv2.imread('dog.png')
img2 = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow("Dog Image", img)
cv2.imshow("Gray Dog Image", img2)

# Wait for infinite time
cv2.waitKey(0)
# Wait for 2500ms
# cv2.waitkey(2500)
cv2.destroyAllWindows()

