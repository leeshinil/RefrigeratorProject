# import 해야할 것
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
		
import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'  # 실제 인셉션 v3 위치
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# 이미지 디렉토리에서 input 데이터 찾아 데이터로 변환
def create_image_lists(image_dir, testing_percentage, validation_percentage):   # 이미지 디렉토리로부터 sub folder들을 분석하고, 그들을 training, testing, validation sets으로 나눈다.
                                                                                # 각각의 label을 위한 이미지 list와 그들의 경로(path)를 나타내는 자료구조(data structure)를 반환.
  if not gfile.Exists(image_dir):                                               # 파일 찾지 못했을때
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}                                                                   # result : 반환할 사전형식 값(이미지 list, 경로)
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]                              # sub_dirs : sub폴더 경로들 받아옴.(리스트 형식)
                                                                                # root directory는 처음에 온다. 따라서 이를 skip한다.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []                                                              # file_list : 파일경로리스트 들어감.
    dir_name = os.path.basename(sub_dir)                                        # dir_name = 파일이름, os.path.basename(sub_dir) : sub_dir = 경로이기때문에 파일이름만 받아옴.
    if dir_name == image_dir:                                                   # root 경로랑 dir_name과 같으면 continue
      continue
    print("Looking for images in '" + dir_name + "'")                           # dir_name 폴더의 파일 받아왔다고 표시함.
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)           # root경로에서 파일이름으로 경로확장(파일의 경로 말해줌.)
      file_list.extend(gfile.Glob(file_glob))                                   # 파일 경로 list에 추가
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      print('WARNING: Folder {} has more than {} images. Some images will '
            'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9A-Z]+', ' ', dir_name)                        # 파일이름이 어떻게 시작하느냐 : A_Z 추가해줌
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      
      hash_name = re.sub(r'_nohash_.*$', '', file_name)                         # 어떤 이미지로 리스트를 만들지 결정할때 파일 이름에 "_nohash_"가 포함되어 있으면 이를 무시할 수 있다.
                                                                                # 그리고 우리는 더많은 파일들이 추가되더라도, 같은 set에 이미 존재하는 파일들이 유지되길 원한다.
                                                                                # 그렇게 하기 위해서는, 우리는 파일 이름 그자체로부터 결정하는 안정적인 방법이 있어야만 한다.
                                                                                # 따라서, 우리는 파일 이름을 hash하고, 이를 이를 할당하는데 사용하는 확률을 결정하는데 사용한다.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {                                                   # image_lists['LA'] 로 검색가능, label_name[1]로 변경해, label_name 이였음.     # 여기!!!!!!!!
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result                                                                 # result : 키를 label_name('A','a','Z','z')으로하는 이중 사전구조 반환

# 주어진 index 대한 이미지 경로 리턴하는 함수
def get_image_path(image_lists, label_name, index, image_dir, category):        # (label대한 전체이미지들, label이름, label내의 인덱스, root 경로, 이미지에 넣을 이름?)
  if label_name not in image_lists:                                             # 전체 이미지들중에 label 이름이 있는지 확인
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]                                         # label_lists : 전체 이미지에서 label_name의 이미지리스트만 가져옴.
  if category not in label_lists:                                               # label 이름을 가진 리스트('a'내에는 'a_001'...있음)중에 카테고리 있는지 확인.
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]                                         # category_list : 레이블 리스트중에 카테고리 리스트만 가져옴
  if not category_list:                                                         # 카테고리 리스트에 값 들어갔는지 확인
    tf.logging.fatal('Label %s has no images in the category %s.',label_name, category)
  mod_index = index % len(category_list)                                        # mod_index : 카테고리 길이를 인덱스로 나눈 나머
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)                       # full_path : root 경로에서 sub경로 확장한 하나 이미지 파일의 경로가 됨. 
  return full_path

# 주어진 index 레이블에 대한 bottleneck 파일 경로 반환
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
  return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'

# 파일에서 텐서플로우 그래프 읽어 리턴함
def create_inception_graph():
  with tf.Graph().as_default() as graph:                                        # 텐서플로우 내의 그래프그려주는 함수 graph라는 이름으로 가져옴.
    model_filename = os.path.join(                                              # 모델경로에 classify_image_graph_def.pb 합쳐
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

# 이미지 대한 추론 실행해 bottleneck 요약 계층 추출
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
  bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

# Inception 그래프가 없다면 다운로드 받음.
def maybe_download_and_extract():
  dest_directory = FLAGS.model_dir                                              # dest_directory : 미리 준비된 데이터셋 경로
  if not os.path.exists(dest_directory):                                        # 경로 없으면 만들어주고,
    os.makedirs(dest_directory)                                                 
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):                                              # 추출하지못할때(데이터셋 없을때)
    def _progress(count, block_size, total_size):                               # 데이터 다운받음.
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
      
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress) 
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

# 디렉토리 경로가 있는지 체크하고 없으면 만든다.
def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)                                                       # 디스크에 없으면 만들어줌.

# 지정된 목록을 이진파일에 씀
def write_list_of_floats_to_file(list_of_floats, file_path):
  s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)                
  with open(file_path, 'wb') as f:
    f.write(s)

# 주어진 파일에서 float 리스트 읽음.
def read_list_of_floats_from_file(file_path):
  with open(file_path, 'rb') as f:
    s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
    return list(s)

# 전역변수 선언
bottleneck_path_2_bottleneck_values = {}

# bottleneck 파일에 저장한다.
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor):
  print('Creating bottleneck at ' + bottleneck_path)                        
  image_path = get_image_path(image_lists, label_name, index,image_dir, category) # 하나 이미지 경로
  if not gfile.Exists(image_path):                                              # 파일 존재x 때
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()                         # 이미지 데이터 읽어옴
  try:
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor
                                                , bottleneck_tensor)
  except:
    raise RuntimeError('Error during processing file %s' % image_path)

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)

# bottleneck 만든다
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
  label_lists = image_lists[label_name]                                         # 내가 원하는 label 리스트 받음
  sub_dir = label_lists['dir']                                                  # sub_dir : label 리스트 들어간 폴더이름 받음
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)                          # sub_dir_path : 그 폴더의 경로
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):                                       # 존재하지 않으면 파일 만들어줌.
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()                                  # 병목파일 읽음
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir
                           , category, sess, jpeg_data_tensor,bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values

# 만든 bottleneck 임시 저장
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(sess, image_lists, label_name, index,
                                 image_dir, category, bottleneck_dir,
                                 jpeg_data_tensor, bottleneck_tensor)
        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')

# 랜덤 혹은 전체 보틀넥 가져와 모델에 집어넣는다
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir
                                  , image_dir, jpeg_data_tensor, bottleneck_tensor):
  class_count = len(image_lists.keys())                                         # class_count : 이미지 리스트 키값의 개수(몇개 클래스로 나뉠수있느냐)
  bottlenecks = []
  ground_truths = []
  filenames = []                                                                # filenames : 랜덤으로 검색한 파일 이름 저장할 곳
  if how_many >= 0:                 
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
      bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir
                                            , category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                              image_index, image_dir, category,
                                              bottleneck_dir, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames

# 왜곡 후 훈련 이미지에 대한 bottleneck 값 검색
def get_random_distorted_bottlenecks(sess, image_lists, how_many, category
                                     , image_dir, input_jpeg_tensor, distorted_image
                                     , resized_input_tensor, bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                         resized_input_tensor,
                                         bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)

# 이미지 처리시에 좌우 변환 등 왜곡 줄지 결정
def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or(random_brightness != 0))

# 지정된 왜곡 적용하는 작업 생성
def add_input_distortions(flip_left_right, random_crop, random_scale,random_brightness):
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0, maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
  precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                  MODEL_INPUT_DEPTH])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min, maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result

# 텐서에 많은 요약 부착
def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# 훈련 위해 새 소프트웨어 맥스와 완전 연결 계층 추가
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)

# 결과 정확성 평가하는 데 필요한 작업 삽입함.
def add_evaluation_step(result_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction

# main단계
def main(_):
  # TensorBoard의 summaries를 write할 directory를 설정한다.(경로설정)
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # pre-trained graph를 생성한다.
  maybe_download_and_extract()                                                    # maybe_download_and_extract : 모델 추출 or 다운로드
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (            # create_inception_graph() : 저장된 그래프 함수 파일로부터 그래프 생성, 반환함.
      create_inception_graph())

  # 폴더 구조를 살펴보고, 모든 이미지에 대한 lists를 생성한다.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0: 
    print('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    print('Only one valid folder of images found at ' + FLAGS.image_dir +
          ' - multiple classes are needed for classification.')
    return -1

  # 커맨드라인 flag에 distortion에 관련된 설정이 있으면 distortion들을 적용한다.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  with tf.Session(graph=graph) as sess:

    if do_distort_images:
      # 우리는 distortion들을 적용할것이다. 따라서 필요한 연산들(operations)을 설정한다.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop,
           FLAGS.random_scale, FLAGS.random_brightness)
    else:
      # 우리는 계산된 'bottleneck' 이미지 summaries를 가지고 있다. 
      # 이를 disk에 캐싱(caching)할 것이다. 
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        bottleneck_tensor)

    # 우리가 학습시킬(training) 새로운 layer를 추가한다.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    # 우리의 새로운 layer의 정확도를 평가(evalute)하기 위한 새로운 operation들을 생성한다.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # 모든 summaries를 합치고(merge) summaries_dir에 쓴다.(write)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # 우리의 모든 가중치들(weights)과 그들의 초기값들을 설정한다.
    init = tf.global_variables_initializer()
    sess.run(init)

    # 커맨드 라인에서 지정한 횟수만큼 학습을 진행한다.
    for i in range(FLAGS.how_many_training_steps):
      # bottleneck 값들의 batch를 얻는다. 이는 매번 distortion을 적용하고 계산하거나,
      # disk에 저장된 chache로부터 얻을 수 있다.
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_data_tensor,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
             bottleneck_tensor)
      # grpah에 bottleneck과 ground truth를 feed하고, training step을 진행한다.
      # TensorBoard를 위한 'merged' op을 이용해서 training summaries을 capture한다.

      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # 일정 step마다 graph의 training이 얼마나 잘 되고 있는지 출력한다.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                        train_accuracy * 100))
        print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                   cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                bottleneck_tensor))
        # validation step을 진행한다.
        # TensorBoard를 위한 'merged' op을 이용해서 training summaries을 capture한다.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
              (datetime.now(), i, validation_accuracy * 100,
               len(validation_bottlenecks)))

    # 트레이닝 과정이 모두 끝났다.
    # 따라서 이전에 보지 못했던 이미지를 통해 마지막 test 평가(evalution)을 진행한다.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
      print('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          print('%70s  %s' % (test_filename,
                              list(image_lists.keys())[predictions[i]]))

    # 학습된 graph와 weights들을 포함한 labels를 쓴다.(write)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')

#########################요기 지정해줘야함################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='D:\\python_D\stuff',                                             # 여기1
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='D:\\python_D\\inception_file\\output_graph8.pb',                            # 여기2
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='D:\\python_D\\inception_file\\output_labels8.txt',                          # 여기3
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='D:\\python_D\\inception_file\\retrain_logs8',                               # 여기4
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=1000,                        # 나는 1000 해야했음                             # 원래 1000이였음
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,                        # 나는 0.001 해야했음
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='D:\\python_D\\inception_file\\imagenet',                                 # 여기5
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='D:\\python_D\\inception_file\\bottleneck8',                               # 여기6 (보틀넥이미지 공유 가)
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

####################################################################################
