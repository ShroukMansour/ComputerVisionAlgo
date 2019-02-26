import cv2

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
exit_loop = False
cropped_img = []

def crop_img(path):
    image = cv2.imread(path)
    oriImage = image.copy()

    def mouse_crop(event, x, y, flags, param):
        global x_start, y_start, x_end, y_end, cropping, exit_loop, cropped_img
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y
        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False  # cropping is finished
            refPoint = [(x_start, y_start), (x_end, y_end)]
            if len(refPoint) == 2:  # when two points were found
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)
                cropped_img = roi
                exit_loop = True

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        cv2.waitKey(1)
        if exit_loop:
            cv2.destroyWindow("image")
            return cropped_img


print("end")
