import picamera
import time
import datetime
def capture():
	with picamera.PiCamera() as camera:

		camera.resolution = (1024,768)
		now = datetime.datetime.now()
		filename = now.strftime('%Y-%m-%d %H:%M:%S')
		camera.start_preview()
		time.sleep(5)
		camera.stop_preview()
		camera.capture('/home/pi/refrigerator/ref/'+filename + '.jpg')
		return filename

if __name__ == '__main__':
	print(capture())
