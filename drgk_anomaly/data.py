from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import cv2
import sys
import threading
import numpy as np
import tensorflow as tf
import logging

# dataset_utils


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        # 'image/class/label': _int64_feature(label),
        # 'image/class/text': _bytes_feature(text),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        pass
        # Create a single Session to run all image coding calls.
        # self._sess = tf.Session()
        # Initializes function that converts PNG to JPEG data.
        # self._png_data = tf.placeholder(dtype=tf.string)
        # image = tf.image.decode_png(self._png_data, channels=3)
        # self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        # Initializes function that decodes RGB JPEG data.
        # self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        # self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        image = tf.image.decode_png(image_data)
        image = tf.image.encode_jpeg(image, format='rgb', quality=100)
        return image
        # return self._sess.run(self._png_to_jpeg,feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        # image = self._sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data: image_data})
        image = tf.image.decode_jpeg(image_data)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    with open(filename, 'rb') as f:
        image_data = f.read()
    if _is_png(filename):
        logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards, output_directory):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s_%s_%.5d-of-%.5d.tfrecord' % (name, name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            image_buffer, height, width = _process_image(filename, coder)
            example = _convert_to_example(filename, image_buffer, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if not counter % 1000:
                logging.info('%s [thread %d]: Processed %d of %d images in thread batch.' %
                             (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        writer.close()
        logging.info('%s [thread %d]: Wrote %d images to %s' %
                     (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    logging.info('%s [thread %d]: Wrote %d images to %d shards.' %
                 (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, output_directory, num_threads=1, num_shards=1):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    # Launch a thread for each batch.
    logging.info('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()
    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, num_shards, output_directory)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    # Wait for all the threads to terminate.
    coord.join(threads)
    logging.info('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir):
    """Build a list of all images files and labels in the data set.
    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.
          data_dir/normal/normal_i.jpg
          data_dir/anormal/anormal_i.jpg
        where 'dog' is the label associated with these images.
    Returns:
      filenames: list of strings; each string is a path to an image file.
    """
    logging.info('Determining list of input files and labels from %s.' % data_dir)
    pattern = os.path.join(data_dir, '*.jpg')
    filenames = tf.gfile.Glob(pattern)
    return filenames


def produce_dataset_from_files(name, source_directory, target_directory):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    filenames = _find_image_files(data_dir=source_directory)
    _process_image_files(name, filenames=filenames, output_directory=target_directory)


def parser(record):
    '''Function to parse a TFRecords example.
    Args:
        record:tfrecord数据中的一个记录
    '''
    # Define here the features you would like to parse
    features = {'image/height': tf.FixedLenFeature((), tf.int64),
                'image/width': tf.FixedLenFeature((), tf.int64),
                'image/colorspace': tf.FixedLenFeature((), tf.string),
                'image/channels': tf.FixedLenFeature((), tf.int64),
                # 'image/class/label': _int64_feature(label),
                # 'image/class/text': _bytes_feature(text),
                'image/format': tf.FixedLenFeature((), tf.string),
                'image/filename': tf.FixedLenFeature((), tf.string),
                'image/encoded': tf.FixedLenFeature((), tf.string)
                }

    # Parse example
    example = tf.parse_single_example(record, features)

    # Decode image
    # 解码后的图象归范到-1.0 ~ 1.0间
    img = tf.image.decode_jpeg(example['image/encoded'])
    img = tf.cast(img, tf.float32)
    img = tf.divide(img, 127.5)
    img = tf.subtract(img, tf.ones_like(img))
    return img


def clear_dir(dir):         # 将 dir 下的内容清理干净
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def _crop_to_patch(image, patch_size=64, patch_stride=32):
    im_height, im_width, _ = image.shape
    h, w, y, x = 0, 0, 0, 0
    patches = []
    while y <= im_height - patch_size:
        x, w = 0, 0
        while x <= im_width - patch_size:
            patch = image[y:y+patch_size, x:x+patch_size, :].copy()
            patches.append(patch)
            x += patch_stride
            w += 1
        y += patch_stride
        h += 1
    return patches, h, w


def _blur(image):
    return cv2.GaussianBlur(image, (11, 11), 0)


def patch_the_image_with_save(source_dir, output_dir, size=64, bluring=False):  # 将某个目录下的jpeg文件，生成切片用于训练

    if os.path.exists(output_dir):
        clear_dir(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene_images_path = os.path.join(source_dir, '*.jpg')
    for _, scene_file in enumerate(tf.gfile.Glob(scene_images_path)):
        # print(scene_file)
        # print(os.path.basename(scene_file))
        scene_file_basename = os.path.splitext(os.path.basename(scene_file))[0]
        print(scene_file_basename)
        scene_image = cv2.imread(scene_file)
        if bluring:
            scene_image = _blur(scene_image)  # 注意！！！
        patches, _, _ = _crop_to_patch(scene_image, patch_size=size, patch_stride=8)  # patch_stride = int(size/2)
        print(len(patches))
        for i in range(len(patches)):
            patch_file = os.path.join(output_dir, scene_file_basename+"_p_{}.jpg".format(i))
            # cv2.imwrite(patch_file,cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))
            cv2.imwrite(patch_file, patches[i])


def patch_the_image_into_batch(imagePath, patch_size, bluring=False):  # 将某个目录下的jpeg文件，生成切片成一批，用于测试
    image = cv2.imread(imagePath)
    if bluring:
        image = _blur(image)  # 注意！
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    patches, h, w = _crop_to_patch(image, patch_size=patch_size, patch_stride=patch_size)
    assert (len(patches)) == h * w
    patches = np.stack(patches, axis=0)
    patches = tf.cast(patches, tf.float32)
    patches = tf.divide(patches, 127.5)
    patches = tf.subtract(patches, tf.ones_like(patches))
    return patches, h, w


def depatch_patches(batch_x, hg, wg):
    # batch_x is a tensor!
    b, h, w, c = batch_x.get_shape().as_list()
    assert b == hg*wg
    if c == 1:
        img = np.zeros((hg*h, wg*w), dtype='float32')
        for i in range(hg):
            for j in range(wg):
                img[i*h:(i+1)*h, j*w:(j+1)*w] = batch_x[i*wg+j, :, :, 0]
    else:
        img = np.zeros((hg*h, wg*w, c), dtype='float32')
        for i in range(hg):
            for j in range(wg):
                img[i*h:(i+1)*h, j*w:(j+1)*w, :] = batch_x[i*wg+j, :, :, :]
    return img


def produce_dataset_from_patches_of_files(name, source_directory, target_directory, patch_size, bluring=False):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    def _to_example(patch, height, width):
        """Build an Example proto for an example.
        Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
        Returns:
        Example proto
        """
        colorspace = 'RGB'
        channels = 3
        image_format = 'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/colorspace': _bytes_feature(colorspace),
            'image/channels': _int64_feature(channels),
            # 'image/class/label': _int64_feature(label),
            # 'image/class/text': _bytes_feature(text),
            'image/format': _bytes_feature(image_format),
            'image/filename': _bytes_feature("NoName"),
            'image/encoded': _bytes_feature(patch)}))
        return example

    filenames = _find_image_files(data_dir=source_directory)
    output_filename = '%s.tfrecord' % (name)
    output_file = os.path.join(target_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    for filename in filenames:
        image = cv2.imread(filename)
        if bluring:
            image = _blur(image)  # 注意！
        patches, h, w = _crop_to_patch(image, patch_size=patch_size, patch_stride=8)
        for patch in patches:
            _, patch_encoded = cv2.imencode('.jpg', patch)
            example = _to_example(patch_encoded.tostring(), h, w)
            writer.write(example.SerializeToString())
    writer.close()


def produce_dataset_from_resized_image(name, source_directory, target_directory, resized_size, bluring=False):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    def _to_example(filename, image_buffer, height, width):
        """Build an Example proto for an example.
        Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
        Returns:
        Example proto
        """
        colorspace = 'RGB'
        channels = 3
        image_format = 'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/colorspace': _bytes_feature(colorspace),
            'image/channels': _int64_feature(channels),
            # 'image/class/label': _int64_feature(label),
            # 'image/class/text': _bytes_feature(text),
            'image/format': _bytes_feature(image_format),
            'image/filename': _bytes_feature(filename),
            'image/encoded': _bytes_feature(image_buffer)}))
        return example

    filenames = _find_image_files(data_dir=source_directory)
    output_filename = '%s.tfrecord' % (name)
    output_file = os.path.join(target_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    for filename in filenames:
        image = cv2.imread(filename)
        image = cv2.resize(image, (resized_size, resized_size), interpolation=cv2.INTER_CUBIC)
        if bluring:
            image = _blur(image)  # 注意！
        _, image_buffer = cv2.imencode('.jpg', image)
        example = _to_example(filename, image_buffer.tostring(), resized_size, resized_size)
        writer.write(example.SerializeToString())
    writer.close()
