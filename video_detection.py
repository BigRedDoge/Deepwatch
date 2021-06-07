import sys
import math
import time
import cv2
import numpy as np
import urllib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from yolov3.yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)

from yolov3.yolov3_tf2.dataset import transform_images
from yolov3.yolov3_tf2.utils import draw_outputs
from lib.VideoStream import VideoStream

flags.DEFINE_string('classes', './yolov3/data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './yolov3/checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './yolov3/data/video.mp4',
                    'path to video file or number for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

URL = "http://129.21.65.167:7777/shot.jpg"

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('Weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('Classes loaded')

    times = []

    #try:
    #    vid = cv2.VideoCapture(int(FLAGS.video))
    #except:
    #    vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(stream.vid().get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.vid().get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(stream.vid().get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        if FLAGS.video == 'webcam':
            #TODO: handle disconnection
            img_ip = urllib.request.urlopen(URL)
            img_arr = np.array(bytearray(img_ip.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
        else:
            # Start thread to display webcam video
            stream = VideoStream(src=int(FLAGS.video)).start()
            frame = stream.read()

        frame_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_in = tf.expand_dims(frame_in, 0)
        frame_in = transform_images(frame_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(frame_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
        frame = cv2.putText(frame, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if FLAGS.output:
            out.write(frame)
 
        cv2.imshow("output", frame)
 
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
