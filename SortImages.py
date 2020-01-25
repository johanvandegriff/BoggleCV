import cv2, os
import numpy as np


INPUT_IMAGES="E:/Letters2/Letters/"
pth2img='E:/Letters2/Letters/0124201552-01.3gp/00/000000000.png'


LENGTH, width, _=cv2.imread(pth2img).shape
print(LENGTH, width)
if LENGTH != width:
    raise(ValueError)

videoletters=os.listdir(INPUT_IMAGES)

MAX_DISP_DIM = 500

def resizeWithAspectRatio(image, maxDispDim, inter=cv2.INTER_AREA):
    h = image.shape[0]
    w = image.shape[1]

    ar = h / w
    #print(ar)
    if h > w:
        nw = maxDispDim
        nh = int(maxDispDim / ar)
    elif w > h:
        nh = maxDispDim
        nw = int(maxDispDim / ar)
    else:
        nh = maxDispDim
        nw = maxDispDim

    dim = None
    (w, h) = image.shape[:2]

    r = nw / float(h)
    dim = (nw, int(w * r))

    return cv2.resize(image, dim, interpolation=inter)

def imshow_fit(name, img, maxDispDim=MAX_DISP_DIM):
    if maxDispDim is not None:
        img = resizeWithAspectRatio(img, maxDispDim)
    cv2.imshow(name, img)



def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def LoadFrame(i,ImageStructure):
    imageName = str(i).zfill(9) + ".png"
    print(3)
    TheseImages = []
    for j in range(25):
        directory = str(j).zfill(2)
        thisimagename = ImageStructure + '/' + directory + '/' + imageName
        print(thisimagename)
        img=cv2.imread(thisimagename)
        TheseImages.append(img)




    return TheseImages







def loadimages(folder):
    L=len(os.listdir(folder + "/00/"))
    print(2)

    for i in range(L):
        relativeDispSize=7
        thisFrame=LoadFrame(i, folder)
        dispSize = LENGTH*relativeDispSize
        B=int(((relativeDispSize-5)/6)*LENGTH)
        image = create_blank(dispSize, dispSize, rgb_color=(255,0,0))
        for index, letter in enumerate(thisFrame):
            xindex=index%5
            yindex=int(index/5)
            xposition= (xindex) * LENGTH+(xindex+1)*B
            yposition=(yindex) * LENGTH+(yindex+1)*B
            image[yposition:yposition+LENGTH, xposition:xposition+LENGTH] = thisFrame[index]

        #image[100:130,100:130]=thisFrame[0]
        imshow_fit("Image", image)
        cv2.waitKey()






for vid in videoletters:
    loadimages(INPUT_IMAGES+vid)
    print(1)

