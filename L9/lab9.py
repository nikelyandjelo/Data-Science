import cv2
import random
from PIL import Image, ImageDraw
def ColorS():
    image = Image.open("Donald.jpg")
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    for i in range(width):
        for j in range(height):
            a = pix[i, j][0]
            b = pix[i, j][1]
            c = pix[i, j][2]
            S = (a + b + c) // 3
            draw.point((i, j), (S, S, S))
    image.save("Donald_S.jpg", "JPEG")
def Nois(factor):
	image = Image.open("Donald_S.jpg")
	draw = ImageDraw.Draw(image)
	width = image.size[0]
	height = image.size[1]
	pix = image.load()
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0] + factor
			b = pix[i, j][1] + factor
			c = pix[i, j][2] + factor
			if (a < 0):
				a = 0
			if (b < 0):
				b = 0
			if (c < 0):
				c = 0
			if (a > 255):
				a = 255
			if (b > 255):
				b = 255
			if (c > 255):
				c = 255
			draw.point((i, j), (a, b, c))
	image.save("Donald_SI.jpg", "JPEG")


img = cv2.imread('Donald.jpg')
cv2.imshow('0',img)
ColorS()
img = cv2.imread('Donald_S.jpg')
cv2.imshow('Donald_S',img)
Nois(70)
img = cv2.imread('Donald_SI.jpg')
cv2.imshow('Donald_SI',img)
image = Image.open("Donald_SI.jpg")
draw = ImageDraw.Draw(image)
width = image.size[0]
height = image.size[1]
pix = image.load()
for i in range(width):
 for j in range(height):
			rand = random.randint(-25, 25)
			a = pix[i, j][0] + rand
			b = pix[i, j][1] + rand
			c = pix[i, j][2] + rand
			if (a < 0):
				a = 0
			if (b < 0):
				b = 0
			if (c < 0):
				c = 0
			if (a > 255):
				a = 255
			if (b > 255):
				b = 255
			if (c > 255):
				c = 255
			draw.point((i, j), (a, b, c))
image.save("Donald_SINois.jpg", "JPEG")
img = cv2.imread('Donald_SINois.jpg')
cv2.imshow('Donald_SINois',img)
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
cv2.imwrite("Donald_SINois-Nois.jpg", dst)
image = Image.open("Donald_SINois-Nois.jpg")
draw = ImageDraw.Draw(image)
width = image.size[0]
height = image.size[1]
pix = image.load()
for i in range(width):
		for j in range(height):
			a = pix[i, j][0] -70
			b = pix[i, j][1] -70
			c = pix[i, j][2] -70
			if (a < 0):
				a = 0
			if (b < 0):
				b = 0
			if (c < 0):
				c = 0
			if (a > 255):
				a = 255
			if (b > 255):
				b = 255
			if (c > 255):
				c = 255
			draw.point((i, j), (a, b, c))
image.save("Donald_SINois-Nois.jpg", "JPEG")
img = cv2.imread('Donald_SINois-Nois.jpg')
cv2.imshow('Donald_SINois-Nois-Irkost',img)

read = 'Donald_SINois-Nois.jpg'
image = cv2.imread(read)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 10,40)
_,contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
cv2.imshow('Contours', image)
cv2.imwrite("Contour1.jpg", image)
