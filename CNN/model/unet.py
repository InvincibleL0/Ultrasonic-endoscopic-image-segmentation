import os
import logging
import time
from datetime import datetime
import tensorflow as tf
from models import Unet
from utils import save_images

import sys
sys.path.append("D:/Code/Intestinal ultrasound/data")
from data import read_tfrecords

import numpy as np
import cv2

class UNet(object):
    def __init__(self, sess, tf_flags):
        self.sess = sess
        self.dtype = tf.float32

        # 模型保存的文件夹：e.g. */model-output
        self.output_dir = tf_flags.output_dir
        # checkpoint文件保存目录 e.g. */model-output/checkpoint
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        # checkpoint文件前缀名
        self.checkpoint_prefix = "model.ckpt"
        self.saver_name = "checkpoint"
        # summary文件保存的目录 e.g. */model-output/summary
        self.summary_dir = os.path.join(self.output_dir, "summary")

        self.is_training = (tf_flags.phase == "train")
        # 初始学习率
        self.learning_rate = 0.00002            #0.0002

        # data parameters
        self.image_w = 512
        self.image_h = 512
        self.image_c = 3

        # 输入大小
        self.input_data = tf.placeholder(self.dtype, [None, self.image_h, self.image_w,self.image_c])
        # mask大小
        self.input_masks = tf.placeholder(self.dtype, [None, self.image_h, self.image_w,self.image_c])

        # 定义学习率占位符
        self.lr = tf.placeholder(self.dtype)

        # train
        if self.is_training:
            # 训练集目录
            self.training_set = tf_flags.training_set
            self.sample_dir = "D:/Code/Intestinal ultrasound/datasets/train_results"

            # 创建summary_dir，checkpoint_dir，sample_dir
            self._make_aux_dirs()

            # 定义 loss，优化器，summary，saver
            self._build_training()

            # 日志文件路径
            log_file = 'Unet.log' #self.output_dir + "/Unet.log"
            logging.basicConfig(filename=log_file,  # 日志文件名
                                level=logging.DEBUG,# 日志级别：只有级别高于DEBUG的内容才会输出
                                format='%(asctime)s [%(levelname)s] %(message)s',  # handler使用指明的格式化字符串:日志时间 日志级别名称 日志信息
                                filemode='a')  # 打开日志文件的模式
            logging.getLogger().addHandler(logging.FileHandler(log_file))   #20191113
            # logging.getLogger()创建一个记录器
            # addHandler()添加一个StreamHandler处理器
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # test
            self.testing_set = tf_flags.testing_set
            # build model
            self.output = self._build_test()

    def _build_training(self):
        """
        定义self.loss,self.opt,self.summary,self.writer,self.saver
        """

        self.output = Unet(name="UNet", in_data=self.input_data, reuse=False)

        # loss. softmax交叉熵函数。  损失函数用来衡量网络准确性，提高预测精度并减少误差，损失函数越小越好。
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.input_masks, logits=self.output))

        # self.loss = tf.reduce_mean(tf.squared_difference(self.input_masks,
        #     self.output))
        # Use Tensorflow and Keras at the same time.
        # self.loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        #     self.input_masks, self.output))

        # 准确率计算  20191111添加
        # tf.equal()返回一个bool值，两参数相等时返回1 , tf.argmax()就是返回最大的那个数值所在的下标 , tf.cast()转换数据类型
        correct_prediction = tf.equal(self.input_masks,self.output)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


        # optimizer
        # 定义Adam优化器,是一个寻找全局最优点的优化算法，引入了二次方梯度校正
        # Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="opt")

        # summary    用来显示标量信息
        tf.summary.scalar("acc", self.acc)      # 20191111添加
        tf.summary.scalar("loss", self.loss)

        # 添加一个操作，代表执行所有summary操作，这样可以避免人工执行每一个summary op。
        self.summary = tf.summary.merge_all()

        # summary and checkpoint        用于将Summary写入磁盘，需要制定存储路径logdir
        # 如果传递了Graph对象，则在Graph Visualization会显示Tensor Shape Information。执行summary op后，将返回结果传递给add_summary()方法即可。
        self.writer = tf.summary.FileWriter(self.summary_dir, graph=self.sess.graph)

        # 最多保存10个最新的checkpoint文件 , tf.train.Saver()用来保存tensorflow训练模型的,默认保存全部参数
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()

    def train(self, batch_size, training_steps, summary_steps, checkpoint_steps, save_steps):
        """
        参数：
        batch_size:每批数据量大小
        training_steps:训练要经过多少迭代步
        summary_steps:每经过多少步就保存一次summary
        checkpoint_steps:每经过多少步就保存一次checkpoint文件
        save_steps:每经过多少步就保存一次图像
        """
        step_num = 0
        # restore last checkpoint e.g. */model-output/checkpoint/model-10000.index
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        # 存在checkpoint文件

        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."

            # 使用最新checkpoint文件restore模型
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.now(),step_num, latest_checkpoint))

        else:
            # 不存在checkpoint文件，初始化模型参数
            self.sess.run(tf.global_variables_initializer())  # init all variables
            logging.info("{}: Init new training".format(datetime.now()))

        # 定义Read_TFRecords类的对象tf_reader
        tf_reader = read_tfrecords.Read_TFRecords(filename=os.path.join(self.training_set,"tfrecords","Unet.tfrecords"),
                                                  batch_size=batch_size, image_h=self.image_h, image_w=self.image_w,
                                                  image_c=self.image_c)


        images, images_masks = tf_reader.read()
        logging.info("{}: Done init data generators".format(datetime.now()))

        # 线程协调器
        # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
        self.coord = tf.train.Coordinator()
        # 使用tf.train.start_queue_runners之后，才会启动填充队列的线程，这时系统就不再“停滞”。
        # 此后计算单元就可以拿到数据并进行计算，整个程序也就跑起来了
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        try:
            # train
            c_time = time.time()
            lrval = self.learning_rate
            for c_step in range(step_num + 1, training_steps + 1):
                # xxxxx个step后，学习率减半
                if c_step % 10000 == 0:
                    lrval = self.learning_rate *0.8

                batch_images, batch_images_masks = self.sess.run([images, images_masks])
                # 实现反向传播需要的参数
                c_feed_dict = {
                    # TFRecord
                    self.input_data: batch_images,
                    self.input_masks: batch_images_masks,
                    self.lr: lrval
                }
                self.sess.run(self.opt, feed_dict=c_feed_dict)

                # save summary
                if c_step % summary_steps == 0:
                    # summary loss
                    c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                    # 写summary文件
                    self.writer.add_summary(c_summary, c_step)

                    e_time = time.time() - c_time
                    logging.info("{}: Iteration={} {}".format(
                        datetime.now(), c_step, self._print_summary(c_summary)))  # self._print_summary(c_summary)：(acc=   )(loss=   )
                    c_time = time.time()  # update time


                # save checkpoint        Iteration是迭代次数，每次迭代更新一次网络结构的参数。
                if c_step % checkpoint_steps == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                                    global_step=c_step)
                    logging.info("{}: Iteration={} Saved checkpoint".format(datetime.now(), c_step))

                # 保存图片
                if c_step % save_steps == 0:
                    # 预测的分割mask和ground truth的mask
                    input_image, output_masks, input_masks = self.sess.run(         # 改了input_iamge,本来是_
                        [self.input_data, self.output, self.input_masks],
                        feed_dict=c_feed_dict)

                    save_images(input_image, output_masks, input_masks,             #input_image本来是None
                                # self.sample_dir：train_results
                                input_path="{}/input_{:04d}.png".format(self.sample_dir, c_step),
                                image_path="{}/train_{:04d}.png".format(self.sample_dir, c_step))
        except KeyboardInterrupt:
            print('Interrupted')
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            # 主线程计算完成，停止所有采集数据的进程
            self.coord.request_stop()
            # 等待其他线程结束
            self.coord.join(threads)
        logging.info("{}: Done training".format(datetime.now()))

    def _build_test(self):
        # network.
        output = Unet(name="UNet", in_data=self.input_data, reuse=False)
        test_prediction = tf.equal(tf.argmax(self.input_data, 1), tf.argmax(output, 1))
        self.acc = tf.reduce_mean(tf.cast(test_prediction, tf.float32))

        # summary    用来显示标量信息
        tf.summary.scalar("acc", self.acc)

        self.summary = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.summary_dir, graph=self.sess.graph)

        # 最多保存10个最新的checkpoint文件 , tf.train.Saver()用来保存tensorflow训练模型的,默认保存全部参数
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()
        # define saver, after the network!
        return output

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def test(self,test_path):

        # 批量测试图片
        image_names = os.listdir(test_path)

        if not os.path.exists(os.path.join(self.testing_set, "test_results")):
            os.makedirs(os.path.join(self.testing_set, "test_results"))

        for image_name in image_names:

            image_file = os.path.join(test_path ,image_name)

            # In tensorflow, test image must divide 255.0.  分成255
            image = cv2.imread(image_file, 1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = np.reshape(cv2.resize(image, (self.image_h, self.image_w)),(1, self.image_h, self.image_w, self.image_c)) / 255.

            print("{}: Done init data generators".format(datetime.now()))

            c_feed_dict = { self.input_data: image }

            output_mask = self.sess.run(self.output, feed_dict=c_feed_dict)

            c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
            logging.info("{}: Test={} {}".format(
                datetime.now(), image_name,
                self._print_summary(c_summary)))

            cv2.imwrite(os.path.join(self.testing_set,"test_results",image_name),np.uint8(output_mask[0].clip(0., 1.) * 255.))



    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _print_summary(self, summary_string):
        # 解析loss summary中的值
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result)
