import cv2 
import time
import os
import redis

url = "rtmp://10.0.11.49/live/demo"
# url = "rtsp://admin:Dou123789@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

temp_folder = "temp"
queue_name = "img_live_demo"


# connect with redis server
img_redis = redis.Redis(host='localhost', port=6379, db=0)
# while True:
#     video_stream = cv2.VideoCapture(url)
video_stream = cv2.VideoCapture(url)
while (video_stream.isOpened()):
    temp_img = "{}.png".format(time.time())
    temp_path = os.path.join(temp_folder, temp_img)
    success, img = video_stream.read()
    if not success: break
    width, heigh, channel = img.shape
    # img = img[:, int(heigh*0.3): int(heigh*0.6), :]
    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    print (temp_path)
    cv2.imwrite(temp_path, img)
    print (img.shape)
    img_redis.publish(queue_name, temp_path)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('img', img)

cv2.destroyAllWindows()
