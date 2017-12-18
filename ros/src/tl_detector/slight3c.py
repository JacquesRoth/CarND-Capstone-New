import cv2
import numpy as np
import math
isCarla = True
RED = 1
GREEN = 2
YELLOW = 3
UNKNOWN = 4
colortxt = ['RED', 'GREEN', 'YELLOW', 'UNKNOWN']
# Integrators (only uint and rint used now)
rint = 0
yint = 0
gint = 0
uint = 5
#from styx_msgs.msg import TrafficLight
global best_light_x, best_light_y, delta_light
best_light_x = 100000.0
best_light_y = 100000.0
delta_light =  100000.0

global rexpected
rexpected = 7
FrameH = 1096
FrameHC = FrameH / 2
FrameW = 1368
FrameWC = FrameW / 2

CameraHeight = 0.5  # Camera height on car
CameraCos =  0.9959 # Uptilt angle cos and sin
CameraSin = -0.09 
CameraPan = 0.06 
CameraF   = 1049.0 #Camera focal length & pixel adjustments
PanError = 0.16
PanPixels = int(CameraF * PanError)
# print("PanPixels", PanPixels) 

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def render_region_of_interest(img, vertices):
    for i in range(0, 4):
        x1, y1 = vertices[i]
        next_vertices = i + 1 if i < 3 else 0
        x2, y2 = vertices[next_vertices]
        cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)


def render_recognized_circles(image, circles, border_color=(0, 0, 0), center_color=(0, 0, 255)):
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], border_color, 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, center_color, 3)


def recognize_light(image, lower_range, upper_range):
    ranged_image = cv2.inRange(image, lower_range, upper_range)
    return ranged_image


def get_hough_circles(weighted_image, hsv_image, debug):
    bestfraction = 0.0
    posColor = UNKNOWN
    blur_img = cv2.GaussianBlur(weighted_image, (15, 15), 0) # (5, 5), 0)
    blur_img = get_canny_edge(blur_img)
    # Start param2 was 7
    minR = int(rexpected * 0.75 - 0.5)
    maxR = int(rexpected * 1.25 + 0.5)
    if minR < 6:  minR = 6
    if maxR > 25: maxR = 25
    circles = cv2.HoughCircles(blur_img, cv2.HOUGH_GRADIENT, 0.5, 41, param1=30, param2=7,\
                               minRadius=minR, maxRadius=maxR)
    if circles == None:
        if debug: print("None H")
        return (None, bestfraction, posColor)
    elif debug: print("Hough circle for lit light found")
    bestcircle = None
    bestcount = -1
    bestindex = 0
    frameH, frameW = blur_img.shape

    #
    # Look at circles returned and pick out the best filled in
    # circle.  Throw away circles with very poor filling.
    # 
    i = 0
    for circle in circles[0,:]:
        x = int(circle[0]+0.5)
        y = int(circle[1]+0.5)
        r = int(circle[2]+0.5)
        if debug: print "x, y, r, frameH, frameW", x, y, r, frameH, frameW
        if (y < r) or y > (frameH-2*r): #6*r):
            i+= 1
            continue
        if (x < r) or x > (frameW-r):
            i+= 1
            continue 
        square_img = weighted_image[y-r:y+r, x-r:x+r]
        count  = np.count_nonzero(square_img)
        fraction = float(count) / (4.0*r*r)
        if debug: print("fraction, count",
                         fraction, count)#, fractionb, countb)
        if (fraction  > 0.2) and (count > 10) and (fraction > bestfraction):
            bestcount = count
            bestcircle = circle
            bestfraction = fraction
            #bestsquare = square_img
            bestindex = i
        i += 1
        if debug: print("Circles examined", i)
    
    if bestcircle == None:
        if debug:
            print ("None F", bestfraction, bestcount, r)
        return (None, bestfraction, posColor)
    
    #
    # Form an image around the suspected traffic light
    #
    x = bestcircle[0]
    y = bestcircle[1]
    r = bestcircle[2]
    top =   int(y - 9.0 * r) #was 11.0
    bot =   int(y + 9.0 * r)
    left =  int(x -  2.0 * r) # was 2.5
    right = int(x +  2.0 * r) # was 2.5
    print "Best Circle was", x, y, r
    print "t, b, l, r, frameH, frameW", top, bot, left, right, frameH, frameW
    deltaH = float(bot - top)
    toporig = top
    if top < 0: top = 0
    if left < 0: left = 0
    if bot >= frameH:
        #top -= frameH - bot
        bot = frameH-1
    if right >= frameW: right = frameW-1
    if (top < bot) and (left < right):
        tl_image = hsv_image[top:bot, left:right]
        lower_frame = np.array([10,7,30]) #was 30]
        upper_frame = np.array([24,200,240])
        maskf = cv2.inRange(tl_image, lower_frame,
                                      upper_frame)
        minr    = int(rexpected*0.3) # was r
        maxr    = int(rexpected*2.5)
        if minr < 6:  minr = 6
        if maxr > 26: maxr = 26
        mindist = int(rexpected*6.0)
        #param2 was 15
        circlesf = cv2.HoughCircles(maskf, cv2.HOUGH_GRADIENT,
        0.5, 12, param1=20, param2=15, minRadius=minr,
                                       maxRadius=maxr) #param2 was 15
        ncirclesf = 0
        if circlesf != None:
            sumy = 0.0
            for circlef in circlesf[0,:]:
                yf = circlef[1]
                yfadj = yf - toporig + top
                sumy += yfadj
                if debug:
                    xf = circlef[0]
                    rf = circlef[2] 
                    print "Circlesf x, y, z, r", xf, yf, rf
                ncirclesf += 1
                if ncirclesf >= 40: break
            if ncirclesf != 0:
                avgy =  sumy/float(ncirclesf)
                avgy /= deltaH
                print "sumy, ncirclesf, avgy, bot, top, deltaH", sumy, ncirclesf, avgy,\
                      bot, top, deltaH 
                if   avgy >= 0.6 : posColor = RED
                elif avgy <= 0.4 : posColor = GREEN
                else                : posColor = YELLOW
            else                    : posColor = UNKNOWN
        if debug:
            print "Tl circles len", ncirclesf
            cv2.imshow('tl_box', maskf)
            print "tl box shape", maskf.shape
    if ncirclesf >= 2:
        if ncirclesf <= 2: posColor = UNKNOWN
        if debug: print "Positional Color", colortxt[posColor-1], avgy
        global best_light_x, best_light_y, delta_light
        delta_light = abs(best_light_x - x) +\
                      abs(best_light_y - y)
        best_light_x = x
        best_light_y = y
        result = [[[bestcircle[0], bestcircle[1],
                    bestcircle[2]]]]
    else: return (None, bestfraction, posColor)
    result = np.array(result)
    if debug: print (bestcircle[2], bestfraction)
    return (result, bestfraction, posColor)

def recognize_red_light(hsv_image):

    lower_red = np.array([15,90,100])
    upper_red = np.array([28,255,255])


    red1 = recognize_light(hsv_image, lower_red, upper_red)
    weighted_img = red1
    #cv2.imshow('red image',weighted_img)
    print "Check for RED"
    result, bestfraction, posColor = get_hough_circles(weighted_img, hsv_image, True)
    if result != None: print('Red light seen')
    else: print('Red light not seen')
    #if posColor == YELLOW: return(result, bestfraction, posColor)
    return (result, bestfraction, posColor)


def recognize_green_light(hsv_image):

    lower_green = np.array([ 40,   80,  90])
    upper_green = np.array([140,  255, 255])
    green1 = recognize_light(hsv_image, lower_green, upper_green)
    lower_green = np.array([0, 0, 240])
    upper_green = np.array([5, 5, 255])
    green2 = recognize_light(hsv_image, lower_green, upper_green)
    green1 = cv2.addWeighted(green1, 1.0, green2, 1.0, 0.0)
    print "Check for GREEN"
    result, bestfraction, posColor = get_hough_circles(green1, hsv_image, True)
    if result !=  None: print('Green light seen')
    else: print('Green light not seen')
    cv2.imshow('Green image', green1)
    return (result, bestfraction, posColor)

def recognize_yellow_light(hsv_image):
    lower_yellow = np.array([15,  90, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow1 = recognize_light(hsv_image, lower_yellow, upper_yellow)
    print "Check for YELLOW"
    result, bestfraction, posColor =  get_hough_circles(yellow1, hsv_image, True)
    if result !=  None: print('Yellow light seen')
    cv2.imshow('Yellow image', yellow1)
    #if posColor == RED: return(result, bestfraction, posColor)
    return (result, bestfraction, posColor)

def get_canny_edge(weighted_img):
    kernel       = np.ones((5,5), np.uint8)
    kernel_erode = np.ones((3,3), np.uint8)

    erode   = cv2.erode(weighted_img, kernel_erode,iterations = 1)
    dilated = cv2.dilate(erode,  kernel)
    dilated = cv2.dilate(dilated, kernel)
    dilated = cv2.dilate(dilated, kernel)
   
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    return cv2.Canny(closed, 50, 70)

def recognize_traffic_lights(image, use_roi=False):
    # Define the region of interest to extract
    image_for_recognizing = image
    if use_roi:
        height, width, _ = image.shape
        roi_vertices = np.array([[(0, 2 * height // 3),
                                  (0, 0),
                                  (width, 0),
                                  (width, 2 * height // 3)]],
                                dtype=np.int32)
        roi = region_of_interest(image, roi_vertices)

        # Render the Region of Interest
        render_region_of_interest(image, roi_vertices[0])
        image_for_recognizing = roi

    # Convert to HSV color space to recognize the traffic light
    hsv_image = cv2.cvtColor(image_for_recognizing,
                             cv2.COLOR_BGR2HSV)

    if not isCarla:
        red_circles, fractionr, posRed    = recognize_red_light   (hsv_image)
    else: red_circles = None
    yellow_circles, fractiony, posYellow = recognize_yellow_light(hsv_image)
    if (yellow_circles is not None) and (red_circles is not None):
        if fractionr > fractiony: yellow_circles = None
        else: red_circles = None
    if red_circles is not None:
        return (RED, red_circles) #TrafficLight.RED

    if yellow_circles is not None:
        return (posYellow, yellow_circles) #TrafficLight.YELLOW

    green_circles, fractiong, posGreen = recognize_green_light(hsv_image)
    if green_circles is not None:
        return (GREEN, green_circles) #TrafficLight.GREEN

    return (UNKNOWN, None) #TrafficLight.UNKNOWN

# Main program

# Read motionc.txt file
file = open("rosbags/motionc.txt")
i = 0
line = file.readline()
mdata = {}
while line:
    mstrings = line.split(',')
    mlist = []
    for string in mstrings:
        mlist += [float(string)]
    mdata[mlist[0]] = mlist
    line = file.readline()
for k in mdata:
    d = mdata[k]
    print (d[0], d[1], d[2], d[3], d[6], d[7], d[8], d[9], d[10]) 
debug = True
kernel = np.ones((5,5),np.float32)/25
for i in range(0, 31, 1):
    print("Frame", i)
    # Position of car
    CarX = mdata[float(i)][1]
    CarY = mdata[float(i)][2]
    CarZ = mdata[float(i)][3]
    Oz   = mdata[float(i)][6]
    Ow   = mdata[float(i)][7]
    cosO = 2.0 * Oz * Ow
    sinO = 2.0 * Ow * Ow - 1.0
    CarO = math.atan2(cosO, sinO)
    #Position of traffic light
    Lx = mdata[float(i)][8]
    Ly = mdata[float(i)][9]
    Lz = mdata[float(i)][10]
    print('Car position', CarX, CarY, CarZ)
    print('Orientation', Oz, Ow)
    print("cosO, sinO", cosO, sinO)
    print('Angle', CarO)
    print('Light position', Lx, Ly, Lz)
    #Translate traffic light position so camera is at (0, 0, 0)
    Lx -= CarX
    Ly -= CarY
    Lz -= CarZ + CameraHeight
    print('shift light', Lx, Ly, Lz)
    
    #Rotate traffic light position by orientation so camera
    # faces along the x axis
    #Lz needs no changes
    #
    cosO = math.cos(CarO+CameraPan)
    sinO = math.sin(CarO+CameraPan)
    Lxt = Lx
    Lx =  Lx * cosO + Ly  * sinO
    Ly =  Ly * cosO - Lxt * sinO
    print('Orientation correction', Lx, Ly, Lz)

    #Rotate traffic light position by camera tilt so camera faces
    #along the x axis
    #Ly needs no changes
    #
    Lxt = Lx
    Lx =  Lx  * CameraCos + Lz * CameraSin
    Lz = -Lxt * CameraSin + Lz * CameraCos
    print("Light translated to", Lx, Ly, Lz)
    #
    # Calculate image position
    #
    Ix = int(FrameWC - CameraF * Ly / Lx)
    Iy = int(FrameHC - CameraF * Lz / Lx)
    #
    # Calculate expected radius of traffic light
    #
    rexpected = CameraF * 0.2 / Lx
    print "Expected TL radius", rexpected
    if   Ix < 0:       Ix = 0
    elif Ix >= FrameW: Ix = FrameW - 1
    if   Iy < 0:       Iy = 0
    elif Iy >= FrameH: Iy = FrameH - 1
    if debug: print("Image x, y =", Ix, Iy)
    RecWC = int(CameraF * 1.5 / Lx)
    RecHC = int(CameraF * 1.5 / Lx)
    if RecWC < 10: RecWC = 10
    if RecHC < 10: RecHC = 10
    RecTop   = Iy - RecHC
    RecBot   = Iy + RecHC
    RecLeft  = Ix - RecWC - PanPixels
    RecRight = Ix + RecWC + PanPixels
    if RecTop   < 0         : RecTop   = 0
    if RecBot   > FrameH - 1:
        RecBot   = FrameH - 1
    if RecLeft  < 0         : RecLeft  = 0
    if RecRight > FrameW - 1: RecRight = FrameW - 1
    #print ("Image to read rosbags/%d.png"%i)
    img = cv2.imread("rosbags/%d.png"%i) #specify filename
    if img == None: continue
    frameH, frameW, layers = img.shape
    #
    # Form cropped image, just a rectangle
    #
    cimg = img[RecTop:RecBot, RecLeft:RecRight]
    colorcode, circle = recognize_traffic_lights(cimg) #img)
    #if (delta_light > 40.0): colorcode = UNKNOWN
    if colorcode == RED:
        if rint != 5: rint += 1
        if uint != 0: uint -= 1
        if gint != 0: gint -= 1
        if yint != 0: yint -= 1
    elif colorcode == YELLOW:
        if yint != 5: yint += 1
        if uint != 0: uint -= 1
        if rint != 0: rint -= 1
        if gint != 0: gint -= 1
    elif colorcode == GREEN:
        if gint != 5: gint += 1
        if uint != 0: uint -= 1
        if rint != 0: rint -= 1
        if yint != 0: yint -= 1
    else:
        if uint != 5: uint += 1
        if rint != 0: rint -= 1
        if gint != 0: gint -= 1
        if yint != 0: yint -= 1

    if   uint > max(rint, gint, yint): color = "UNKNOWN"
    elif rint > max(uint, gint, yint): color = "RED"
    elif yint > max(uint, gint, rint): color = "YELLOW"
    else: color = "GREEN"

    if (circle != None):
        #print (img.shape)
        x = int(circle[0][0][0]+0.5)
        y = int(circle[0][0][1]+0.5)
        r = int(circle[0][0][2]+0.5)
        if debug:
            print("delta_light_x,y", delta_light)
        render_recognized_circles(cimg, circle)
        print("Circle x, y", x, y)
    # Draw rectangle around traffic light position
    if Ix != None:
        cv2.rectangle(img,(RecLeft,RecTop),
                          (RecRight,RecBot),(0,255,0),2)
    cv2.putText(img, color+"  "+str(i),(30,50),\
        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),4)
    #cv2.imshow('crop', cimg)
    cv2.imshow('frame',img)
    cv2.imwrite("rosbags/%dout.png"%i, img)
    #cv2.imshow('maskr',maskr)
    #cv2.imshow('masky',masky)
    #cv2.imshow('maskg',maskg)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
cv2.waitKey(0) 
cv2.destroyAllWindows()
