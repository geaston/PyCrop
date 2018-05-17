import cv2 as cv
import glob2 as glob
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
imageFile = 'test2.jpg'
images = glob.glob("*.jpg")  # get all jpegs in the folder
for image in images:  # loop through the jpegs
    img = cv.imread(image)  # open the image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # detect face
    for (x, y, w, h) in faces:  # loop through the face tuple
        halfW = w / 2  # cut image width in half
        halfH = h / 2  # cut image height in half
        newX = x - halfW  # set new x to move left half the image width
        if newX < 0:
            newX = 0  # if new x location is off image pane set to 0
        else:
            newX = x - halfW
        newY = y - halfH  # set new y location to move up half the image width
        if newY < 0:
            newY = 0  # if new y location is off image pane set to 0
        else:
            newY = y - halfH
        newW = w*1.5  # set new image width to be 1.5 times larger to add same buffer on right as left.
        newH = h*1.5  # set new image height to be 1.5 times larger to add same buffer on bottom as top.
        print(int(newX), int(newY))
        # cv.rectangle(img, (int(newX), int(newY)), (x+int(newW), y+int(newH)), (255, 0, 0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[int(newY):y+int(newH), int(newX):x+int(newW)].copy()
        '''adds all the new xy coords to make the new crop area on face detection'''
        maxsize = (250, 250)  # set image size to default image pop up size
        imRes = cv.resize(roi_color, maxsize, interpolation=cv.INTER_AREA)  # resize image with new width,height
    print(image)
    cv.imwrite(image, imRes, [int(cv.IMWRITE_JPEG_QUALITY), 75])
    cv.imshow('img', imRes)
    print(imRes.shape)
    cv.waitKey(0)
    cv.destroyAllWindows()


