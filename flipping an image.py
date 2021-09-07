import cv2, os

path = 'IDRiD_032.jpg'

originalImage = cv2.imread(path)
originalImage = cv2.resize(src=originalImage,dsize=(800, 800) )

flipVertical = cv2.flip(src=originalImage, flipCode=0)

flipHorizontal = cv2.flip(src=originalImage, flipCode=1) # flipcode=1 for horizontal flip

flipBoth = cv2.flip(src=originalImage, flipCode=-1) # hflip and vflip both

cv2.imshow('Original image', originalImage)
cv2.imshow('Flipped vertical image', flipVertical)
cv2.imshow('Flipped horizontal image', flipHorizontal)
cv2.imshow('Flipped both image', flipBoth)

cv2.waitKey(0)
cv2.destroyAllWindows()