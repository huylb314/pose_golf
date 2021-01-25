import redis
import cv2
import time

queue_name = "img_live_demo"

# connect with redis server as Bob
img_redis = redis.Redis(host='localhost', port=6379, db=0)
img_sub = img_redis.pubsub()
# subscribe to classical music
img_sub.subscribe(queue_name)
length = img_redis.llen(queue_name)
for msg in img_sub.listen():
    print (length)
    mess = msg['data']
    if mess != 1:
        img_path = mess.decode('UTF-8')
        print(mess.decode('UTF-8'))
        img = cv2.imread(img_path)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
        cv2.imshow('img_2', img)
cv2.destroyAllWindows()