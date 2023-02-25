import cv2
import numpy as np
import keras
from PIL import Image


WINDOW_DIM = (500, 400, 3)
DRAW_DIM = (400, 400, 3)
THICKNESS = 25

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

model = keras.models.load_model("my model.h5") # loading our model
print("model loading successful")

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=THICKNESS)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=THICKNESS)


img = np.zeros(WINDOW_DIM, np.uint8)   #this is the image?
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)
prediction = "None"
while(1):
    cv2.imshow('test draw',img)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.imwrite("test.png", img)
        break
    elif k == ord("q"):
        img = img[100:500, 0:400] #crop_img = img[y:y+h, x:x+w]
        print(img.shape)
        cv2.imwrite("testog.png", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28), cv2.INTER_NEAREST)

        #img = cv2.bitwise_not(img)  # invert image
        #ret,thresh1 = cv2.threshold(img,254,255,cv2.THRESH_BINARY) #not sure if this does anything


        #print(img.shape)
        cv2.imwrite("test.png", img)

        img = np.expand_dims(img, axis=0)
        #print(img.shape)
        res = model.predict(img)
        prediction = np.argmax(res)
        print(prediction)
        #print(img)

        img = np.zeros(WINDOW_DIM, np.uint8)
    cv2.putText(img, 'Prediction: ' + str(prediction), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)


cv2.destroyAllWindows()




# import tkinter as tk
# def myfunction(event):
#     x, y = event.x, event.y
#     if canvas.old_coords:
#         x1, y1 = canvas.old_coords
#         canvas.create_line(x, y, x1, y1)
#     canvas.old_coords = x, y
#
# root = tk.Tk()
#
# canvas = tk.Canvas(root, width=400, height=400)
# canvas.pack()
# canvas.old_coords = None
#
# root.bind('<B1-Motion>', myfunction)
# root.mainloop()
#
#
# from PIL import ImageGrab
#
# def getter(widget):
#     x=root.winfo_rootx()+widget.winfo_x()
#     y=root.winfo_rooty()+widget.winfo_y()
#     x1=x+widget.winfo_width()
#     y1=y+widget.winfo_height()
#     ImageGrab.grab().crop((x,y,x1,y1)).save("file path here")
