"""Video frame helpers shared by the processing pipeline."""

import cv2


def resize_frame(frame, width=640):
    height, original_width = frame.shape[:2]

    if original_width <= width:
        return frame

    new_height = int(height * width / original_width)

    return cv2.resize(frame, (width, new_height))


def encode_frame(frame):
    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        return None

    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
