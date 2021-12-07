# run with python object_size.py --image images/example_01.png --width 0.955

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2.cv2 as cv2

BRIGHTNESS = 10
WIDTH = 3
HEIGHT = 4

path = '../assets/example2.jpeg'
cap = cv2.VideoCapture(0)
cap.set(BRIGHTNESS, 10)
cap.set(WIDTH, 1920)
cap.set(HEIGHT, 1080)
scale = 3
paperWidth = 210 * scale
paperHeight = 297 * scale


def midpoint(point_a, point_b):
    return (point_a[0] + point_b[0]) * 0.5, (point_a[1] + point_b[1]) * 0.5


class MeasureObject:

    def __init__(self, know_width, is_camera=False):
        self.is_camera = is_camera
        if self.is_camera:
            success, self.image = cap.read()
        else:
            self.image = cv2.imread(path)
        self.blurred_image = None
        self.edged_image = None
        self.image_contours = None
        self.know_width = know_width
        self.pixels_per_metric = None

    # load the image, convert it to grayscale, and blur it slightly
    # image = cv2.imread(args["image"])
    def apply_image_effects(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        self.__setattr__('blurred_image', blur)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    def detect_edges(self, blurred_image):
        edged = cv2.Canny(blurred_image, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        self.__setattr__('edged_image', edged)

    # find contours in the edge map
    def find_contours(self):
        image_contours = cv2.findContours(self.edged_image.copy(), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        image_contours = imutils.grab_contours(image_contours)

        # sort the contours from left-to-right
        (image_contours, _) = image_contours.sort_contours(image_contours)
        self.__setattr__('image_contours', image_contours)

    # loop over the contours individually
    def handler(self):

        for contour in self.image_contours:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(contour) < 100:
                continue

            # compute the rotated bounding box of the contour
            orig = self.image.copy()
            box = cv2.minAreaRect(contour)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (top_left, top_right, bottom_right, bottom_left) = box
            (top_left_top_right_x, top_left_top_right_y) = midpoint(top_left, top_right)
            (bottom_left_bottom_right_x, bottom_left_bottom_right_y) = midpoint(bottom_left, bottom_right)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and bottom-right
            (top_left_bottom_left_x, top_left_bottom_left_y) = midpoint(top_left, bottom_left)
            (top_right_bottom_right_x, top_right_bottom_right_y) = midpoint(top_right, bottom_right)
            # draw the midpoints on the image
            cv2.circle(orig, (int(top_left_top_right_x), int(top_left_top_right_y)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(bottom_left_bottom_right_x), int(bottom_left_bottom_right_y)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(top_left_bottom_left_x), int(top_left_bottom_left_y)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(top_right_bottom_right_x), int(top_right_bottom_right_y)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(top_left_top_right_x), int(top_left_top_right_y)), (int(bottom_left_bottom_right_x),
                                                                                    int(bottom_left_bottom_right_y)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(top_left_bottom_left_x), int(top_left_bottom_left_y)), (int(top_right_bottom_right_x),
                                                                                        int(top_right_bottom_right_y)),
                     (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            distance_a = dist.euclidean((top_left_top_right_x, top_left_top_right_y),
                                        (bottom_left_bottom_right_x, bottom_left_bottom_right_y))
            distance_b = dist.euclidean((top_left_bottom_left_x, top_left_bottom_left_y),
                                        (top_right_bottom_right_x, top_right_bottom_right_y))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if self.pixels_per_metric is None:
                self.pixels_per_metric = distance_b / self.know_width

            # compute the size of the object
            dimension_a = distance_a / self.pixels_per_metric
            dimension_b = distance_b / self.pixels_per_metric
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimension_a),
                        (int(top_left_top_right_x - 15), int(top_left_top_right_y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimension_b),
                        (int(top_right_bottom_right_x + 10), int(top_right_bottom_right_y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            # show the output image
            return orig

    def get_image(self):

        while True:
            cv2.imshow("Image", self.handler())
            cv2.waitKey(0)
