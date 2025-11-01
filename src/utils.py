import cv2
import numpy as np


def draw_text(img, text, org=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
              scale=0.8, color=(0, 255, 0), thickness=2, bgcolor=(0, 0, 0)):
    """Draw text with background for readability."""
    x, y = org
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 5, y - h - 5), (x + w + 5, y + 5), bgcolor, -1)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return img


def resize_for_display(frame, width=1280):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    r = width / float(w)
    dim = (int(w * r), int(h * r))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


