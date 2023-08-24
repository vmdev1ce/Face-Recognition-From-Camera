# import
import cv2

cap = cv2.VideoCapture(0)

#capOpened
print(cap.isOpened())

#capRead
while cap.isOpened():
    ret_flag, img_camera = cap.read()

    print("height: ", img_camera.shape[0])
    print("width:  ", img_camera.shape[1])
    print('\n')

    cv2.imshow("camera", img_camera)
    #The data delay of each frame is 1ms, the delay is 0, and the static frame is read
    k = cv2.waitKey(1)
    #Press 's' to save the screenshot
    if k == ord('s'):
        cv2.imwrite("test.jpg", img_camera)
    # Press 'q' to quit
    if k == ord('q'):
        break
# Release all cameras
cap.release()
# delete all created windows
cv2.destroyAllWindows()
