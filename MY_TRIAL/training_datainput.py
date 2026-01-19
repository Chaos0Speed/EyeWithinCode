import subprocess
subprocess.run(["rm", "-rf", "./captured/anchor", "./captured/positive"], check=True)
subprocess.run(["mkdir", "-p", "./captured/anchor", "./captured/positive"], check=True)

import cv2
import os

cam = cv2.VideoCapture(0)
cv2.namedWindow("Image Capture")
anchor_img_counter = 0
positive_img_counter = 0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = frame[200:450, 170:420, :]
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Image Capture", frame)

    k = cv2.waitKey(10)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k % 256 == 97:
        # 'a' pressed
        img_name = f"anchor_image_{anchor_img_counter}.png"
        cv2.imwrite(os.path.join('./captured/anchor',img_name), frame)
        print(f"{img_name} saved!")
        anchor_img_counter += 1

    elif k % 256 == 112:
        # 'p' pressed
        img_name = f"positive_image_{positive_img_counter}.png"
        cv2.imwrite(os.path.join('./captured/positive',img_name), frame)
        print(f"{img_name} saved!")
        positive_img_counter += 1

cam.release()
cv2.destroyAllWindows()