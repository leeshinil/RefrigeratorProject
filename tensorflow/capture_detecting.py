# -*- coding: utf-8 -*-
"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""
import numpy as np
import tensorflow as tf
import picamera
import time
import datetime
import pymysql

def capture():
        with picamera.PiCamera() as camera:
                camera.resolution = (1024,768)
                now = datetime.datetime.now()
                filename = now.strftime('%Y-%m-%d %H:%M:%S')
                camera.start_preview()
                time.sleep(5) #5seconds
                camera.stop_preview()
                camera.capture('/home/pi/refrigerator/ref/'+filename + '.jpg')
		return filename

file = capture()
imagePath = '/home/pi/refrigerator/ref/' + file  + '.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = '/home/pi/refrigerator/output_graph9.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/home/pi/refrigerator/output_labels9.txt'                                   # 읽어들일 labels 파일 경로

def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(imagePath = imagePath):
    answer = None
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        # 여기서 5개 가져옴
        # -> break하면 끝남.
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
	    name = human_string
	    break;
	database(name, file)
        answer = labels[top_k[0]]

        return answer


#dbinsert
def database(name,file):
	conn =  pymysql.connect(host='117.17.142.135',port=3306, user='root', password='1234', db='refrigerator')
	curs = conn.cursor()

	#find onion's expiration date
	sql = "select * from expired where name = %s"
	curs.execute(sql,name)
	rows = curs.fetchall()

	for i in rows:
		ex=i[1]

#	print(name)
#	print(file)
#	print(ex)

	#input today, expiration date
	val = (name, file, ex)
	sql2 = "insert into refrigerator (name, inputdate, expiration) values (%s,%s,%s)"
	curs.execute(sql2,val)
	conn.commit()

if __name__ == '__main__':

    	run_inference_on_image()
