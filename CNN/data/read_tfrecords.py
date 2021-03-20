
import tensorflow as tf

class Read_TFRecords(object):
    def __init__(self, filename,batch_size=2,image_h=512, image_w=512, image_c=3,
                 num_threads=8, capacity_factor=3, min_after_dequeue=1000):
        """
        filename: TFRecords file path.
        num_threads: TFRecords file load thread.
        capacity_factor: capacity.
        """
        self.filename = filename
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.num_threads = num_threads
        self.capacity_factor = capacity_factor
        self.min_after_dequeue = min_after_dequeue

    def read(self):
        # read a TFRecords file, return tf.train.batch/tf.train.shuffle_batch object.
        # 从TFRecords文件中读取数据
        # 第1步：需要用tf.train.string_input_producer生成一个文件名队列。
        filename_queue = tf.train.string_input_producer([self.filename])

        # 第2步：调用tf.TFRecordReader创建读取器
        reader = tf.TFRecordReader()
        # 读取文件名队列，返回serialized_example对象
        key, serialized_example = reader.read(filename_queue)

        # 第3步：调用tf.parse_single_example操作将Example协议缓冲区(protocol buffer)解析为张量字典
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               "image_raw": tf.FixedLenFeature([], tf.string),
                                               "image_label": tf.FixedLenFeature([], tf.string),
                                           })

        # 第4步：对图像张量解码并进行一些处理resize,归一化...
        ## tensorflow里面提供解码的函数有两个，tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，得到图像的像素值
        image_raw = tf.image.decode_png(features["image_raw"], channels=self.image_c, name="decode_image")
        image_label = tf.image.decode_png(features["image_label"], channels=self.image_c, name="decode_image")

        if self.image_h is not None and self.image_w is not None:
            image_raw = tf.image.resize(image_raw, [self.image_h, self.image_w],
                                               method=tf.image.ResizeMethod.BILINEAR)
            image_label = tf.image.resize(image_label, [self.image_h, self.image_w],
                                                 method=tf.image.ResizeMethod.BILINEAR)

        # 像素值类型转换为tf.float32，归一化
        image_raw = tf.cast(image_raw, tf.float32) / 255.0  # convert to float32
        image_label = tf.cast(image_label, tf.float32) / 255.0  # convert to float32

        # tf.train.batch/tf.train.shuffle_batch object.
        # tf.train.shuffle_batch()该函数将会使用一个队列，函数读取一定数量的tensors送入队列，将队列中的tensor打乱，
        # 然后每次从中选取batch_size个tensors组成一个新的tensors返回出来
        # 参数：
        # tensors：要入队的tensor列表
        # batch_size:表示进行一次批处理的tensors数量
        # capacity:为队列的长度，建议capacity的取值如下：min_after_dequeue + (num_threads + a small safety margin) * batch_size
        # min_after_dequeue:意思是队列中，做dequeue（取数据）的操作后，线程要保证队列中至少剩下min_after_dequeue个数据。
        #                   如果min_after_dequeue设置的过少，则即使shuffle为True，也达不到好的混合效果,过大则会占用更多的内存
        # num_threads:决定了有多少个线程进行入队操作，如果设置的超过一个线程，它们将从不同文件不同位置同时读取，可以更加充分的混合训练样本,设置num_threads的值大于1,使用多个线程在tensor_list中读取文件
        # allow_smaller_final_batch(False)：当allow_smaller_final_batch为True时，如果队列中的张量数量不足batch_size，将会返回小于batch_size长度的张量，如果为False，剩下的张量会被丢弃
        # Using asynchronous queues

        # 第5步：tf.train.shuffle_batch将训练集打乱，每次返回batch_size份数据
        input_data, input_masks = tf.train.shuffle_batch([image_raw, image_label],
                                                         batch_size=self.batch_size,
                                                         capacity=self.min_after_dequeue + self.capacity_factor * self.batch_size,
                                                         min_after_dequeue=self.min_after_dequeue,
                                                         num_threads=self.num_threads,
                                                         name='images')

        return input_data, input_masks  # return list or dictionary of tensors.
