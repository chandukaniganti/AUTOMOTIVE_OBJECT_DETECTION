import cv2
import numpy as np


def detect_lanes(frame, debug=False):
    """Simple lane detection using Canny + HoughLinesP. Returns annotated frame."""
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    mask = np.zeros_like(edges)
    # region of interest polygon (focus on lower half)
    polygon = np.array([[
        (0, h),
        (w, h),
        (w, int(h * 0.6)),
        (0, int(h * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if debug:
        return img, edges
    return img
