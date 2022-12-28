# Code written by Lorenzo Lezza
# USE Python 3.7.0 and install these modules
import cv2
import mediapipe as mp
import math
import numpy as np
import glob

# Change this to your project folder path
projDir = "C:\Your\Project\Folder\\"
# Change this to your photo folder path (must not be empty)
phDir = "C:\Your\Photos\\"
# Choose fps (15 recommended)
fps = 15

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
# Adjust maximum number of faces if you took group photos
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, static_image_mode=True)

img_names = glob.glob(projDir + "/*.jpg")
first_img = cv2.imread(img_names[0])
(h, w, c) = first_img.shape
resolution = (w, h)
center = (int(resolution[0]/2), int(resolution[1]/2))

fnf = 0

############### FUNCTIONS ##############

# Returns the distance between two points
def point_dist(a, b):
    xa, ya = a
    xb, yb = b
    diffx = abs(xa-xb)
    diffy = abs(ya-yb)
    return math.sqrt((diffx**2) + (diffy**2))

# Converts a face landmark to coordinates
def lm2coord(lm):
    w, h = resolution
    return (int(lm.x*w), int(lm.y*h))

# Moves an image overlapping point p to point t
def to_target(img, p, t):
    xc, yc = t
    xp, yp = p
    offx = abs(xp - xc) if (xp <= xc) else -abs(xp - xc)
    offy = abs(yp - yc) if (yp <= yc) else -abs(yp - yc)

    M = np.float32([[1, 0, offx], [0, 1, offy]])
    out = cv2.warpAffine(img, M, resolution)
    return out

# Shrinks an image based on the 'scale' value (between 0 and 1)
def shrink(img, pivot, scale):
    M = cv2.getRotationMatrix2D(pivot, 0, scale)
    out = cv2.warpAffine(img, M, resolution)
    return out

# Calculates the degrees needed to rotate the image so that the eyes are in a straight line. Rotates around the pivot
def rotate(img, pivot, ple, pre):
    deg = -90 - math.atan2(ple[0] - pre[0], ple[1] - pre[1]) * 180 / math.pi

    M = cv2.getRotationMatrix2D(pivot, deg, 1)
    out = cv2.warpAffine(img, M, resolution)
    return out

# Checks if a point is at the center of the image (inside the second third of the height and width)
def at_center(p):
    px, py = p
    rw, rh = resolution
    return (rw/3 <= px <= rw*2/3) and (rh/3 <= py <= rh*2/3)

# Returns the face that is closest to the center among a given set
def c_closest(faces):
    mindist = float("inf")
    minface = False

    if (faces):
        for face in faces:
            nose = lm2coord(face.landmark[4])
            if at_center(nose):
                dist = point_dist(center, nose)
                if (dist < mindist):
                    mindist = dist
                    minface = face

    return minface

# (UTILITY) Draws a green dot in the specified position of the image
def drawp(i, p):
    out = cv2.circle(i, p, 5, (0, 255, 0), 5)

    cv2.imshow("drawp", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


############### FIND SMALLEST CENTER FACE ##############
# This is needed to resize every photo according to the one in which your face is the most distant to the camera.
# Why? So that no photo will be cropped out of the screen!

sm_eyedist = float("inf")

# This is the position every photo will be aligned to, the left eye of the smallest face found.
leyepos = "NO VALUE"

for filename in img_names:

    print("Checking smallest face in image", filename)

    img = cv2.imread(filename)
    img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = faceMesh.process(imgRGB)

    faces = result.multi_face_landmarks

    if(faces):
        print("Detected", len(faces), "faces")

    face = c_closest(faces)

    if (face):
        nose = lm2coord(face.landmark[4])
        leye = lm2coord(face.landmark[133])
        reye = lm2coord(face.landmark[362])

        dist = point_dist(leye, reye)

        if (dist < sm_eyedist):
            sm_filename = filename
            sm_eyedist = dist
            leyepos = leye
    else:
        print("No faces found!")

print("#### Smallest face is in file", sm_filename)
print("#### Alignment position is at", leyepos)

############### BUILD VIDEO ##############

video = cv2.VideoWriter(projDir + "/Timelapse.mp4", 0x7634706d, fps, resolution)

for filename in img_names:
    print("Processing img " + filename)

    img = cv2.imread(filename)
    img = cv2.resize(img, resolution)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = faceMesh.process(imgRGB)

    face = c_closest(result.multi_face_landmarks)

    if (face):

        nose = lm2coord(face.landmark[4])
        leye = lm2coord(face.landmark[133])
        reye = lm2coord(face.landmark[362])

        # shrink
        eyedist = point_dist(leye, reye)
        img = shrink(img, leye, sm_eyedist/eyedist)

        # to target
        img = to_target(img, leye, leyepos)

        # rotate
        img = rotate(img, leye, leye, reye)

        video.write(img)
    else:
        print("Skipped image",filename,"because no faces were found")
        fnf += 1
print("Number of images with no faces found:", fnf)
video.release()
print("Video was rendered in",projDir)