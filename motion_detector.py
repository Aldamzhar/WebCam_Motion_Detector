import cv2, time, pandas
from datetime import datetime

first_frame = None 
video = cv2.VideoCapture(0)
status_list = [None, None]
times = []

df = pandas.DataFrame(columns=["Start", "End"])

while True:
    # if number is given, then video is captured from webcam, if string with filename then from video from computer
    check, frame = video.read() # boolean and numpy array which shows the pixels array to depict the first frame of video, image
    
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0) 
    
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_delta,None,iterations=2) # smoothing threshold frame, bigger iterations, smoother frame

    cnts,_ = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            
    status_list.append(status)

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())    

    cv2.imshow("Capturing",gray) # creates the window with the first frame
    cv2.imshow("Delta_frame", delta_frame)
    cv2.imshow("Threshold delta", thresh_frame)
    cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1) # 1 millisecond
    
    if key == ord("q"):
        if status == 1:
            times.append(datetime.now())
        break
    
for i in range(0,len(times),2):
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)    
    
df.to_csv("Times.csv")
    
video.release() # releases the window to actually output it to us
cv2.destroyAllWindows()