import cv2, os
import numpy as np


INPUT_IMAGES="E:/Letters2/Letters/"
pth2img='E:/Letters2/Letters/0124201552-01.3gp/00/000000000.png'
KEYS={
    'a':0x61,
    'b':0x62,
    'c':0x63,
    'd':0x64,
    'e':0x65,
    'f':0x66,
    'g':0x67,
    'h':0x68,
    'i':0x69,
    'j':0x6a,
    'k':0x6b,
    'l':0x6c,
    'm':0x6d,
    'n':0x6e,
    'o':0x6f,
    'p':0x70,
    'q':0x71,
    'r':0x72,
    's':0x73,
    't':0x74,
    'u':0x75,
    'v':0x76,
    'w':0x77,
    'x':0x78,
    'y':0x79,
    'z':0x7a,
    'A':0x41,
    'B':0x42,
    'C':0x43,
    'D':0x44,
    'E':0x45,
    'F':0x46,
    'G':0x47,
    'H':0x48,
    'I':0x49,
    'J':0x4a,
    'K':0x4b,
    'L':0x4c,
    'M':0x4d,
    'N':0x4e,
    'O':0x4f,
    'P':0x50,
    'Q':0x51,
    'R':0x52,
    'S':0x53,
    'T':0x54,
    'U':0x55,
    'V':0x56,
    'W':0x57,
    'X':0x58,
    'Y':0x59,
    'Z':0x5a,
    '1':0x31,
    '2':0x32,
    '3':0x33,
    '4':0x34,
    '5':0x35,
    '6':0x36,
    '7':0x37,
    '8':0x38,
    '9':0x39,
    '0':0x30,
    'enter':0xd,
    'backspace':0x20,

}




LENGTH, width, _=cv2.imread(pth2img).shape
#print(LENGTH, width)
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
    #print(3)
    TheseImages = []
    for j in range(25):
        directory = str(j).zfill(2)
        thisimagename = ImageStructure + '/' + directory + '/' + imageName
        #print(thisimagename)
        img=cv2.imread(thisimagename)
        TheseImages.append(img)




    return TheseImages

def mkBoardSortFrame(thisFrame):
    relativeDispSize = 7

    dispSize = LENGTH * relativeDispSize
    B = int(((relativeDispSize - 5) / 6) * LENGTH)
    image = create_blank(dispSize, dispSize, rgb_color=(255, 0, 0))
    for index, letter in enumerate(thisFrame):
        xindex = index % 5
        yindex = int(index / 5)
        xposition = (xindex) * LENGTH + (xindex + 1) * B
        yposition = (yindex) * LENGTH + (yindex + 1) * B
        image[yposition:yposition + LENGTH, xposition:xposition + LENGTH] = thisFrame[index]
    return image

def loadimages(folder):
    L=len(os.listdir(folder + "/00/"))
    #print(2)

    # for i in range(L):
    i=0
    while i < L:
        thisFrame = LoadFrame(i, folder)
        image=mkBoardSortFrame(thisFrame)
        #image[100:130,100:130]=thisFrame[0]
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow("Image",MAX_DISP_DIM,MAX_DISP_DIM)
        imshow_fit("Image", image)
        key=cv2.waitKey()&0xff
        print(hex(key))
        if key == 0x08: #backspace
            i -= 1
        else:
            i += 1






for vid in videoletters:
    loadimages(INPUT_IMAGES+vid)
    #print(1)

