import tensorflow as tf
from model import unet
import os


def main(argv):
    # tf.app.flags.FLAGS接受命令行传递参数或者tf.app.flags定义的默认参数
    tf_flags = tf.flags.FLAGS

    # gpu config.
    # tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置
    config = tf.ConfigProto()

    # tf提供了两种控制GPU资源使用的方法，第一种方式就是限制GPU的使用率:
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用50%显存
    # 第二种是让TensorFlow在运行过程中动态申请显存，需要多少就申请多少:
    config.gpu_options.allow_growth = True

    if tf_flags.phase == "train":
        # 使用上面定义的config设置session
        with tf.Session(config=config) as sess:
            # when use queue to load data, not use with to define sess
            # 定义Unet模型
            train_model = unet.UNet(sess, tf_flags)
            # 训练Unet网络，参数：batch_size,训练迭代步......
            train_model.train(tf_flags.batch_size, tf_flags.training_steps,
                              tf_flags.summary_steps, tf_flags.checkpoint_steps, tf_flags.save_steps)

    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = unet.UNet(sess, tf_flags)

            # test阶段:加载checkpoint文件的数据给模型参数初始化
            test_model.load(tf_flags.checkpoint)

            test_model.test(os.path.join(tf_flags.testing_set,"test"))

            print("Saved test files successfully !")

if __name__ == '__main__':
    # tf.flags可以定义一些默认参数，相当于接受python文件命令行执行时后面给的参数。
    # tf.flags.DEFINE_xxx()就是添加命令行的optional argument（可选参数），而tf.flags.FLAGS可以从对应的命令行参数取出参数。
    # 可以不用反复修改源代码中的参数，直接在命令行中进行参数的设定。
    tf.flags.DEFINE_string("output_dir", "D:/Code/Intestinal ultrasound/model-output","checkpoint and summary directory.")
    tf.flags.DEFINE_string("training_set", "D:/Code/Intestinal ultrasound/datasets","dataset path for training.")
    tf.flags.DEFINE_string("testing_set", "D:/Code/Intestinal ultrasound/datasets","dataset path for testing one image pair.")
    tf.flags.DEFINE_string("checkpoint", None, "checkpoint name for restoring.")
    tf.flags.DEFINE_string("phase", "train", "model phase: train/test.")  # 通过更改train/test来进行训练/测试
    ####################################################################################################################
    tf.flags.DEFINE_integer("batch_size", 2,"batch size for training.")            #batch_size就是更新梯度中使用的样本数
    tf.flags.DEFINE_integer("training_steps", 30000,"total training steps.")
    tf.flags.DEFINE_integer("summary_steps", 100,"summary period.")
    tf.flags.DEFINE_integer("checkpoint_steps", 500,"checkpoint period.")
    tf.flags.DEFINE_integer("save_steps", 500,"checkpoint period.")

    tf.app.run(main=main)
