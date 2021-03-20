import tensorflow as tf

def Unet(name,in_data,reuse=False):
    # reuse=False : 不共享变量
    assert in_data is not None           # 确定输入非空
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        conv1_1 = tf.layers.conv2d(in_data, 64, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # 因为padding="SAME",舍去crop，直接merge
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)

        conv2_1 = tf.layers.conv2d(pool1, 128, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        conv3_1 = tf.layers.conv2d(pool2, 256, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2)

        conv4_1 = tf.layers.conv2d(pool3, 512, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        drop4 = tf.layers.dropout(conv4_2)
        pool4 = tf.layers.max_pooling2d(drop4, 2, 2)

        conv5_1 = tf.layers.conv2d(pool4, 1024, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.layers.conv2d(conv5_1, 1024, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        drop5 = tf.layers.dropout(conv5_2)

        # 上采样可改为双线性插值
        up6_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop5)
        up6 = tf.layers.conv2d(up6_1, 512, 2, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge6 = tf.concat([drop4, up6], axis=3)
        conv6_1 = tf.layers.conv2d(merge6, 512, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        up7_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        up7 = tf.layers.conv2d(up7_1, 256, 2, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge7 = tf.concat([conv3_2, up7], axis=3)
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        up8_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        up8 = tf.layers.conv2d(up8_1, 128, 2, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge8 = tf.concat([conv2_2, up8], axis=3)
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        up9_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2)
        up9 = tf.layers.conv2d(up9_1, 64, 2, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        merge9 = tf.concat([conv1_2, up9], axis=3)
        conv9_1 = tf.layers.conv2d(merge9, 64, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv9_2 = tf.layers.conv2d(conv9_1, 64, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # conv9_3 = tf.layers.conv2d(conv9_2, 2, 3, padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv10 = tf.layers.conv2d(conv9_2, 3, 1, activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer())
                                     # 输出3通道图片      sigmoid一般用于二分类任务，softmax一般用于多分类任务
        return conv10
