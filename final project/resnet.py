import numpy as np
import os
import tensorflow as tf
import urllib2
import matplotlib.pyplot as plt

from datasets import imagenet, dataset_utils
from nets import resnet_v1
from preprocessing import vgg_preprocessing


url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
checkpoints_dir = '/tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

slim = tf.contrib.slim

image_size = resnet_v1.resnet_v1_50.default_image_size

with tf.Graph().as_default():
    img = open('test.jpg')
    image_string = img.read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.to_float(image)
    processed_images  = tf.expand_dims(image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1.resnet_v1_50(processed_images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
        slim.get_model_variables('resnet_v1_50'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        writer = tf.summary.FileWriter('/nfs/stak/students/w/wangxiny/workspace/models/slim/', sess.graph)
        writer.add_graph(sess.graph)
        
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index], names[index]))