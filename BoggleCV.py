import cv2, math, os, json, traceback, io
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

IMG_DIR = '../cascademan/categories/5x5/images'
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CONTOUR_THICKNESS = 2
MAX_DISP_DIM = 500

VIDEO_INPUT = "../Boggle-Videos/Boggle5x5/edited/speedup.mp4"
VIDEO_OUT_DIR = "../Boggle-Videos/Boggle5x5/"
VIDEO_ENCODER = "H264"
VIDEO_FPS = 30

#a generic error
class BoggleError(Exception):
    def __init__(self, arg):
        self.strerror = arg
        self.args = {arg}


def four_point_transform(image, pts, size):
    w = size
    h = size
    ntl = [0, 0]
    ntr = [w, 0]
    nbr = [w, h]
    nbl = [0, h]
    location = np.float32(pts)
    newLocation = np.float32([nbl, nbr, ntr, ntl])
    M = cv2.getPerspectiveTransform(location, newLocation)
    unwarpedImage = cv2.warpPerspective(image, M, (w, h))
    return unwarpedImage

def resizeWithAspectRatio(image, maxDispDim, inter=cv2.INTER_AREA):
    w, h, _ = image.shape
    ar = w / h
    print(ar)
    if w > h:
        nw = maxDispDim
        nh = int(maxDispDim / ar)
    elif h > w:
        nh = maxDispDim
        nw = int(maxDispDim / ar)
    else:
        nh = maxDispDim
        nw = maxDispDim

    dim = None
    (h, w) = image.shape[:2]

    r = nw / float(w)
    dim = (nw, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def imshow_fit(name, img, maxDispDim=MAX_DISP_DIM):
    if maxDispDim is not None:
        img = resizeWithAspectRatio(img, maxDispDim)
    cv2.imshow(name, img)

def findBestCtr(contours):
    bestArea = 0
    bestCtr = None
    for i, ctr in enumerate(contours):
        area = cv2.contourArea(ctr)
        # perimeter = cv2.arcLength(ctr, True)
        if area > bestArea:
            bestArea = area
            bestCtr = ctr
    return bestCtr


def angleEveryFew(ctr, step):
    angles = []
    # dists = []
    prev = ctr[-1]
    for i in range(0, len(ctr), step):
        curr = ctr[i]
        px, py = prev[0]
        cx, cy = curr[0]
        # angle = (px-cx)/(py-cy)
        angle = math.atan2(cy - py, cx - px)
        angles.append(angle)
        # dist = math.hypot(cx-px, cy-py)
        # dists.append(dist)
        prev = curr
    return angles


def angleAvg(angles):
    x = 0
    y = 0
    for angle in angles:
        x += math.cos(angle)
        y += math.sin(angle)
    return math.atan2(y, x)


def angleDiffAbs(angle1, angle2):
    return abs(((angle1 - angle2 + math.pi) % (2 * math.pi)) - math.pi)


def runningAvg(angles, history):
    result = []
    for i in range(len(angles)):
        # avg = 0
        vals2 = []
        for j in range(history):
            val = angles[int(i + j - history / 2) % len(angles)]
            vals2.append(val)
            # avg += val
        # avg /= history
        avg = angleAvg(vals2)
        result.append(avg)
    return result


def diffAbs(vals):
    diffs = []
    prev = vals[-1]
    for i in range(0, len(vals), 1):
        curr = vals[i]
        diff = angleDiffAbs(prev, curr) * 10
        diffs.append(diff)
        prev = curr
    return diffs


def debounce(bools, history):
    result = []
    for i in range(len(bools)):
        total = 0
        for j in range(history):
            val = bools[int(i + j - history / 2) % len(bools)]
            total += val
        result.append(int(total >= history / 2))
    return result


def findGaps(diffs):
    wasGap = True
    seamI = 0
    for i, diff in enumerate(diffs):
        isGap = diff
        if not isGap and not wasGap:
            seamI = i
            break
        wasGap = isGap

    xs = range(len(diffs))
    xs = [x for x in xs]
    xs_a = xs[seamI:]
    xs_a.extend(xs[:seamI])
    diffs_a = diffs[seamI:]
    diffs_a.extend(diffs[:seamI])

    xs2 = []
    diffs2 = []
    gapsStart = []
    gapsStartY = []
    gapsEnd = []
    gapsEndY = []
    wasGap = None
    gapwidth = 0
    for x, diff in zip(xs_a, diffs_a):
        isGap = diff
        if wasGap is None:
            wasGap = isGap
        if isGap:
            xs2.append(x)
            diffs2.append(diff)
            gapwidth += 1
        if isGap and not wasGap:
            gapsStart.append(x)
            gapsStartY.append(.5)
        if not isGap and wasGap:
            gapsEnd.append(x)
            gapsEndY.append(gapwidth)
            gapwidth = 0
        wasGap = isGap
    return gapsStart, gapsStartY, gapsEnd, gapsEndY, diffs2, xs2
    # gaps = [i for i in zip(gapsStart, gapsEnd, gapsEndY)]
    # return gaps, diffs, xs2


def top4gaps(gaps):
    def length_sort(gap):
        return -gap[2]

    gaps2 = sorted(gaps, key=length_sort)
    gaps2 = gaps2[:4]
    return [i for i in gaps if i in gaps2]


def invertGaps(gaps):
    segments = []
    prev = gaps[-1]
    for curr in gaps:
        segments.append((prev[1], curr[0]))
        prev = curr
    return segments


def findSidePoints(segments, ctr, step):
    sidePoints = []
    for seg in segments:
        if seg[1] > seg[0]:
            sidePoints.append(ctr[seg[0] * step:seg[1] * step])
        else:
            sidePoints.append(ctr[seg[0] * step:, :seg[1] * step])
    return sidePoints


def getEndVals(arr, fraction):
    if fraction >= 0.5: return arr
    l = len(arr)
    keep = int(fraction * l)
    keep = max(keep, 1)
    return np.concatenate((arr[:keep], arr[l - keep:]))


def fitSidePointsToLines(sidePoints):
    lines = []
    for sp in sidePoints:
        xs = np.zeros(len(sp), int)
        ys = np.zeros(len(sp), int)
        for i, pt in enumerate(sp):
            x, y = pt[0] #TODO if pt is empty
            xs[i] = x
            ys[i] = y
        lines.append(np.polyfit(xs, ys, 1))
    return lines


def findCorners(lines):
    points = []

    prev = lines[-1]
    for curr in lines:
        a1, b1 = prev
        a2, b2 = curr

        x = (b2 - b1) / (a1 - a2)
        y = np.polyval(curr, x)
        #if math.isnan(x): x = 0 #TODO
        #if math.isnan(y): y = 0
        points.append((int(x), int(y)))
        prev = curr
    return points


def drawLinesAndPoints(image, lines, points):
    width = image.shape[1]

    for l in lines:
        y1 = int(np.polyval(l, 0))
        y2 = int(np.polyval(l, width - 1))
        cv2.line(image, (0, y1), (width - 1, y2), RED, CONTOUR_THICKNESS)

    for point in points:
        cv2.circle(image, point, 4, BLUE, 3)


def contourPlot(xs, xs2, angles, anglesAvg, diffs, diffs2, gapsStart, gapsStartY, gapsEnd, gapsEndY2, normalPlots):
    fig = plt.figure(figsize=(8,10))
    ax1 = fig.add_subplot(111)

    # ax1.scatter(xs, dists, s=10, c='b', marker="s", label="dists")
    ax1.scatter(xs, angles, s=10, c='r', marker="o", label="angles")
    ax1.scatter(xs, anglesAvg, s=10, c='b', marker="o", label="anglesAvg")
    ax1.scatter(xs, diffs, s=10, c='g', marker="o", label="diffs")
    ax1.scatter(xs2, diffs2, s=10, c='m', marker="s", label="diffs2")
    ax1.scatter(gapsStart, gapsStartY, s=10, c='c', marker="o", label="gapsStart")
    if gapsEndY2 is not None:
        ax1.scatter(gapsEnd, gapsEndY2, s=10, c='y', marker="o", label="gapsEnd")
    plt.legend(loc='best')
    if normalPlots:
        plt.show(block=False)
        return None
    else:
        return plotToImg()

def waitForKey():
    while True:
        key = cv2.waitKey(0)
        print("key", key)
        if key == 27:  # esc
            cv2.destroyAllWindows()
            quit()
        if key == ord(" ") or key == ord("q"):
            break
    cv2.destroyAllWindows()
    plt.close('all')



#https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
#window: np.ones (flat), np.hanning, np.hamming, np.bartlett, np.blackman
def smooth(x, window_len=11, window=np.hanning):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = window(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def findRowsOrCols(img, doCols, smoothFactor, ax):
    smoothFactor = int(smoothFactor * img.shape[0])
    print("smoothFactor", smoothFactor)
    
    if doCols:
        title = "colSum"
        imgSum = cv2.reduce(img, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32S)
        imgSum = imgSum[0]
    else:
        #row sum
        title = "rowSum"
        imgSum = cv2.reduce(img, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32S)
        imgSum = imgSum.reshape(len(imgSum))
    
    imgSumSmooth = smooth(imgSum, smoothFactor*2)

    #https://qingkaikong.blogspot.com/2018/07/find-peaks-in-data.html
    #peaks_positive, _ = scipy.signal.find_peaks(imgSumSmooth, height=200, threshold = None, distance=60)
    dips, props = scipy.signal.find_peaks(-imgSumSmooth, height=(None,None), distance=30, prominence=(None,None))
    
    #threshold=(None,None), 
    #, plateau_size=(None,None)
    
    print(props)

    prs = props["prominences"]
    if len(prs) < 6:
        top_6_dips = dips
        #print ("!!!! less than 6 dips")
        #raise BoggleError("less than 6 dips")
    else:
        prsIdx = sorted(range(len(prs)), key=lambda i: prs[i], reverse=True)
        print(prsIdx)
        prsIdx = prsIdx[:6]
        print(prsIdx)
        top_6_dips = [p for i,p in enumerate(dips) if i in prsIdx]

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    
    q = [i for i in range(len(imgSumSmooth))]
    
    ax.plot(q, imgSumSmooth, 'b-', linewidth=2, label="smooth")
    ax.plot(np.linspace(smoothFactor,len(imgSumSmooth)-smoothFactor, len(imgSum)), imgSum, 'r-', linewidth=1, label=title)

    #ax.plot(
        #[q[p] for p in peaks_positive],
        #[imgSumSmooth[p] for p in peaks_positive],
        #'ro', label = 'positive peaks')
    
    ax.plot(
        [q[p] for p in dips],
        [imgSumSmooth[p] for p in dips],
        'go', label='dips')
    
    ax.plot(
        [q[p] for p in top_6_dips],
        [imgSumSmooth[p] for p in top_6_dips],
        'c.', label='top 6 dips')
    
    ax.legend(loc='best')
        
    #return top_6_dips
    print("before clip", top_6_dips)
    top_6_dips_scaled = [np.clip(0, p-smoothFactor, len(imgSum)-1) for p in top_6_dips]
    return top_6_dips_scaled

def plotToImg():
    #https://stackoverflow.com/questions/5314707/matplotlib-store-image-in-variable
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    #https://stackoverflow.com/questions/11552926/how-to-read-raw-png-from-an-array-in-python-opencv
    file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    #img_data_cvmat = cv.fromarray(img_data_ndarray) #  convert to old cvmat if needed

    return img_data_ndarray


def findBoggleBoard(image, normalPlots=True, harshErrors=False):
    resultImages = {}
    
    debugimage = image.copy()

    # maskThresholdMin = (108, 28, 12)
    # maskThresholdMax = (125, 255, 241)
    # maskThresholdMin = (108, 28, 6)
    # maskThresholdMax = (130, 255, 241)
    maskThresholdMin = (108, 28, 6)
    maskThresholdMax = (144, 255, 241)
    size = max(image.shape)
    print("size", size)
    blurAmount = int(.02 * size)
    blurAmount = (blurAmount, blurAmount)
    # blurThreshold = 80
    blurThreshold = 40
    contourApprox = cv2.CHAIN_APPROX_NONE
    # contourApprox = cv2.CHAIN_APPROX_SIMPLE
    # contourApprox = cv2.CHAIN_APPROX_TC89_L1
    # contourApprox = cv2.CHAIN_APPROX_TC89_KCOS

    hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvimg, maskThresholdMin, maskThresholdMax)

    maskblur = cv2.blur(mask, blurAmount)
    # maskblur = cv2.threshold(maskblur, 80, 255, cv2.THRESH_BINARY_INV)
    maskblur = cv2.inRange(maskblur, (blurThreshold,), (255,))
    contours, hierarchy = cv2.findContours(maskblur, cv2.RETR_LIST, contourApprox)
    # print(hierarchy) #TODO
    debugmask = cv2.cvtColor(maskblur, cv2.COLOR_GRAY2BGR)
    
    bestCtr = findBestCtr(contours)
    # bestCtr = cv2.convexHull(bestCtr)

    step = 10
    avgWindow = 0.1
    debounceFactor = .05
    
    angles = angleEveryFew(bestCtr, step)

    xs = range(len(angles))

    anglesAvg = runningAvg(angles, int(avgWindow * len(angles)))
    diffs = diffAbs(anglesAvg)
    
    avgDiff = np.mean(diffs)
    print(avgDiff)
    
    binDiffs = [int(i > avgDiff) for i in diffs]
    binDiffs = debounce(binDiffs, int(len(binDiffs) * debounceFactor))
    gapsStart, gapsStartY, gapsEnd, gapsEndY, diffs2, xs2 = findGaps(binDiffs)
    gaps = [i for i in zip(gapsStart, gapsEnd, gapsEndY)]
    
    #scale for viewing on the plot
    gapsEndY2 = gapsEndY
    if len(gapsEndY) > 0:
        q = max(gapsEndY)
        if q > 6: gapsEndY2 = [i * 6 / q for i in gapsEndY]

    contourPlotImg = contourPlot(xs, xs2, angles, anglesAvg, diffs, diffs2, gapsStart, gapsStartY, gapsEnd, gapsEndY2, normalPlots)

    if contourPlotImg is not None:
        resultImages["contourPlotImg"] = contourPlotImg
    
    #draw the contours for debugging
    for img in debugmask, debugimage:
        cv2.drawContours(img, contours, -1, RED, CONTOUR_THICKNESS)
        cv2.drawContours(img, [bestCtr], -1, BLUE, CONTOUR_THICKNESS)
    
    if len(gaps) < 4:
        print("!!!! less than 4 gaps")
        if harshErrors:
            raise BoggleError("less than 4 gaps")
        contourPlotImg = contourPlot(xs, xs2, angles, anglesAvg, diffs, diffs2, gapsStart, gapsStartY, gapsEnd, None, normalPlots)

        if contourPlotImg is not None:
            resultImages["contourPlotImg"] = contourPlotImg
        
        resultImages["debugmask"] = debugmask
        resultImages["debugimage"] = debugimage
        return resultImages
    
    endFraction = 0.01
    
    gaps = top4gaps(gaps)
    segments = invertGaps(gaps)
    sidePoints = findSidePoints(segments, bestCtr, step)
    sidePoints = [getEndVals(sp, endFraction) for sp in sidePoints]
    print("sidepoints len", len(sidePoints[0]))
    
    for img in debugmask, debugimage:
        cv2.drawContours(img, sidePoints, -1, YELLOW, CONTOUR_THICKNESS)
    
    lines = fitSidePointsToLines(sidePoints)
    points = findCorners(lines)
    drawLinesAndPoints(debugimage, lines, points)
    drawLinesAndPoints(debugmask, lines, points)

    npPoints = np.array(points)
    size = 300
    warpedImage = four_point_transform(image, npPoints, size)
        
    warpgray = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
    
    resultImages["debugmask"] = debugmask
    resultImages["debugimage"] = debugimage
    #resultImages["warpgray"] = warpgray
    resultImages["warpedImage"] = warpedImage
    
    smoothFactor = .05
    
    #fig= plt.figure(figsize=(6,3))
    fig, axs = plt.subplots(2, figsize=(8,10))
    #fig.suptitle('imgSum')
    
    rowSumLines = findRowsOrCols(warpgray, False, smoothFactor, axs[0])
    print("rows", rowSumLines)
    colSumLines = findRowsOrCols(warpgray, True, smoothFactor, axs[1])
    print("cols", colSumLines)
    
    
    if normalPlots:
        plt.show(block=False)
    else:
        resultImages["imgSumPlotImg"] = plotToImg()


    if len(rowSumLines) < 6 or len(colSumLines) < 6:
        print("!!!! not enough grid lines")
        if harshErrors:
            raise BoggleError("not enough gridlines")
        return resultImages
    
    #fix the outermost lines of the board
    h1 = rowSumLines[2] - rowSumLines[1]
    h2 = rowSumLines[3] - rowSumLines[2]
    h3 = rowSumLines[4] - rowSumLines[3]
    h = max(h1, h2, h3)
    
    newCSL0 = colSumLines[1] - h
    if newCSL0 > colSumLines[0]:
        colSumLines[0] = newCSL0
    newCSL5 = colSumLines[4] + h
    if newCSL5 < colSumLines[5]:
        colSumLines[5] = newCSL5
    
    w1 = colSumLines[2] - colSumLines[1]
    w2 = colSumLines[3] - colSumLines[2]
    w3 = colSumLines[4] - colSumLines[3]
    w = max(w1, w2, w3)
    
    newRSL0 = rowSumLines[1] - w
    if newRSL0 > rowSumLines[0]:
        rowSumLines[0] = newRSL0
    newRSL5 = rowSumLines[4] + w
    if newRSL5 < rowSumLines[5]:
        rowSumLines[5] = newRSL5

    print("rows2", rowSumLines)
    print("cols2", colSumLines)

    #just display
    plt.figure(figsize=(10,10))
    i = 1
    for y in range(5):
        for x in range(5):
            plt.subplot(5,5,i)
            i += 1
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            minX = colSumLines[x]
            maxX = colSumLines[x+1]
            minY = rowSumLines[y]
            maxY = rowSumLines[y+1]
            crop_img = warpgray[minY:maxY, minX:maxX]
            plt.imshow(crop_img, cmap=plt.cm.gray)
    if normalPlots:
        plt.show(block=False)
    else:
        diceImg1 = plotToImg()
        resultImages["diceImg1"] = diceImg1

    letterResize = 30
    #make square, resize, display, and save to an array
    letterImgs = []
    plt.figure(figsize=(10,10))
    i = 1
    for y in range(5):
        letterImgRow = []
        for x in range(5):
            plt.subplot(5,5,i)
            i += 1
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            minX = colSumLines[x]
            maxX = colSumLines[x+1]
            minY = rowSumLines[y]
            maxY = rowSumLines[y+1]
            w = maxX - minX
            h = maxY - minY
            #print("w,h 1:", w, h)
            d = abs(w - h)
            if d > 0:
                if int(d/2) == d/2:
                    #even difference
                    d1 = d2 = int(d/2)
                else:
                    #odd difference
                    d1 = int((d-1)/2)
                    d2 = int((d+1)/2)
                if w > h:
                    #wider than it is tall
                    minX += d1
                    maxX -= d2
                else:
                    #taller than it is wide
                    minY += d1
                    maxY -= d2
            #print("w,h 2:", maxX-minX, maxY-minY)
            crop_img = warpgray[minY:maxY, minX:maxX]
            letterImg = cv2.resize(crop_img, (letterResize,letterResize), interpolation=cv2.INTER_AREA)
            plt.imshow(letterImg, cmap=plt.cm.gray)
            letterImgRow.append(letterImg)
        letterImgs.append(letterImgRow)
    if normalPlots:
        plt.show(block=False)
    else:
        diceImg2 = plotToImg()
        resultImages["diceImg2"] = diceImg2
    
    return resultImages

def findAndShowBoggleBoard(imgDir, imgFilename):
    imgPath = imgDir + "/" + imgFilename

    #if True:
    try:
        print(imgPath)
        image = cv2.imread(imgPath)
        #imgs = findBoggleBoard(image, normalPlots=True, harshErrors=False)
        imgs = findBoggleBoard(image, normalPlots=False, harshErrors=True)
        for title, img in imgs.items():
            #print("title, img:", title, img)
            #cv2.imshow(title, img)
            imshow_fit(title, img)
        waitForKey()
    except Exception as e:
        #print(str(e))
        print(traceback.format_exc())
        print("=== hit enter to continue ===")
        input()
        print("=== continuing ===")


def processVideo():
    vid = cv2.VideoCapture(VIDEO_INPUT)
    if (vid.isOpened() == False):
        print("Error opening video stream or file")
    ret, frame = vid.read()
    height, width, _ = frame.shape
    resolution = (width, height)
    print("frame size: ", width, "x", height)
    #maskVidout = cv2.VideoWriter(VIDEO_OUT_DIR + maskVidName, cv2.VideoWriter_fourcc(*VIDEO_ENCODER), VIDEO_FPS, resolution)
    
    CLIP_OUT = VIDEO_OUT_DIR + "out1.mp4"
    boggleVidout = cv2.VideoWriter(CLIP_OUT, cv2.VideoWriter_fourcc(*VIDEO_ENCODER), VIDEO_FPS, (300, 300))
    
    #augmentedClipout = cv2.VideoWriter(VIDEO_OUT_DIR + augmentedClip, cv2.VideoWriter_fourcc(*VIDEO_ENCODER), VIDEO_FPS, resolution)
    
    errors = {}
    
    frameNum = 1
    while (vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()
        # #pVid=[pVid,frame]
        if ret == True:
            print("FRAME #", frameNum)
            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            try:
                imgs = findBoggleBoard(frame, normalPlots=False, harshErrors=True)
                imshow_fit("warpedImage", imgs["warpedImage"])
                boggleVidout.write(imgs["warpedImage"])
            except Exception as e:
                print("DROP FRAME #", frameNum)
                error = str(e)
                errors[frameNum] = error
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
        frameNum += 1

    # When everything done, release the video capture object

    vid.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    json.dump(errors, open(CLIP_OUT+'.json', 'w'), indent=2)


def processImages():
    # problems = ["00030.jpg", "00012.jpg"]
    problems = ["00177.jpg", "00176.jpg", "00174.jpg", "00169.jpg", "00167.jpg", "00166.jpg", "00153.jpg"]
    #for img in problems:
        #findAndShowBoggleBoard(IMG_DIR, img)

    #findAndShowBoggleBoard(IMG_DIR, "00168.jpg")
    #findAndShowBoggleBoard(IMG_DIR, "00173.jpg")
    #findAndShowBoggleBoard(IMG_DIR, "00161.jpg")
    
    #not enough grid lines (to test error handling):
    #findAndShowBoggleBoard(IMG_DIR, "00157.jpg")
    #findAndShowBoggleBoard(IMG_DIR, "00156.jpg")


    images = os.listdir(IMG_DIR)
    images.sort()
    images.reverse()
    print(images)
    for img in images:
        if not img in problems:
            findAndShowBoggleBoard(IMG_DIR, img)


if __name__ == "__main__":
    processImages()
