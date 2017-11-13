# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import time
import gdal


class improcess():
    def __init__(self):
        pass

    # no geo information
    def write_tiff(self, file_outpath, input_arr, out_width, out_height, out_band):
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(file_outpath, out_width, out_height, out_band, gdal.GDT_UInt16)
        if out_band == 1:
            dst_ds.GetRasterBand(1).WriteArray(input_arr)
        elif out_band > 1:
            for i in range(out_band):
                dst_ds.GetRasterBand(i + 1).WriteArray(input_arr[i])

    # carry geo information
    def writeTiff(self, im_data, im_width, im_height, im_bands, path, im_geotrans=None, im_proj=None):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
            # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        if (dataset != None) & (im_geotrans != None) & (im_proj != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset

    '''
     def largen_label_provide(filein, fileout):
         impro = improcess()
         file_com = '/media/zb/ml/raw_data/1/preliminary1_raw_data/tinysample.tif'
         im_tiny = gdal.Open(file_com)

         im = gdal.Open(filein)
         im = im.ReadAsArray()

         impro.writeTiff(im, 15106, 5106, 1, fileout, [0, 12, 0, 0, 0, -12], im_tiny.GetProjection())
     '''

    # split feature img
    def adaptive_split_img(self, filein, split_img_path, split_size):
        img = gdal.Open(filein)

        # (4, 5106, 15106)
        width = img.RasterXSize
        height = img.RasterYSize

        rows = height / split_size + 1  # hang
        cols = width / split_size + 1  # lie

        nums = 0
        for i in range(rows):
            for j in range(cols):
                if (i == rows - 1) and (j != cols - 1):
                    yheight = height - split_size * i

                    arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=split_size, ysize=yheight)
                    zero_arr = np.zeros((4, split_size - yheight, split_size))
                    tmp_arr = np.concatenate((arr, zero_arr), axis=1)

                    filename = split_img_path + str(nums) + '.tif'
                    self.write_tiff(filename, tmp_arr, split_size, split_size, 4)
                    nums += 1

                elif (j == cols - 1) and (i != rows - 1):
                    xwidth = width - split_size * j

                    arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=xwidth, ysize=split_size)
                    zero_arr = np.zeros((4, split_size, split_size - xwidth))
                    tmp_arr = np.concatenate((arr, zero_arr), axis=2)

                    filename = split_img_path + str(nums) + '.tif'
                    self.write_tiff(filename, tmp_arr, split_size, split_size, 4)
                    nums += 1

                elif (i == rows - 1) and (j == cols - 1):
                    xwidth = width - split_size * j
                    yheight = height - split_size * i

                    zero_arr1 = np.zeros((4, yheight, split_size - xwidth))
                    zero_arr2 = np.zeros((4, split_size - yheight, split_size))
                    arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=xwidth, ysize=yheight)
                    arr1 = np.concatenate((arr, zero_arr1), axis=2)
                    tmp_arr = np.concatenate((arr1, zero_arr2), axis=1)

                    filename = split_img_path + str(nums) + '.tif'
                    self.write_tiff(filename, tmp_arr, split_size, split_size, 4)
                    nums += 1

                else:
                    split_arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=split_size,
                                                ysize=split_size)
                    filename = split_img_path + str(nums) + '.tif'
                    self.write_tiff(filename, split_arr, split_size, split_size, 4)
                    nums += 1

    # split label img
    def adaptive_split_label(self, filein, split_label_path, split_size):
        img = gdal.Open(filein)

        # (4, 5106, 15106)
        width = img.RasterXSize
        height = img.RasterYSize

        rows = height / split_size + 1  # hang
        cols = width / split_size + 1  # lie

        nums = 0
        for i in range(rows):
            for j in range(cols):
                if (i == rows - 1) and (j != cols - 1):
                    yheight = height - split_size * i

                    arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=split_size, ysize=yheight)
                    zero_arr = np.zeros((split_size - yheight, split_size))
                    tmp_arr = np.concatenate((arr, zero_arr), axis=0)

                    filename = split_label_path + str(nums) + '.tif'
                    self.write_tiff(filename, tmp_arr, split_size, split_size, 1)
                    nums += 1

                elif (j == cols - 1) and (i != rows - 1):
                    xwidth = width - split_size * j

                    arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=xwidth, ysize=split_size)
                    zero_arr = np.zeros((split_size, split_size - xwidth))
                    tmp_arr = np.concatenate((arr, zero_arr), axis=1)

                    filename = split_label_path + str(nums) + '.tif'
                    self.write_tiff(filename, tmp_arr, split_size, split_size, 1)
                    nums += 1

                elif (i == rows - 1) and (j == cols - 1):
                    xwidth = width - split_size * j
                    yheight = height - split_size * i

                    zero_arr1 = np.zeros((yheight, split_size - xwidth))
                    zero_arr2 = np.zeros((split_size - yheight, split_size))
                    arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=xwidth, ysize=yheight)
                    arr1 = np.concatenate((arr, zero_arr1), axis=1)
                    tmp_arr = np.concatenate((arr1, zero_arr2), axis=0)

                    filename = split_label_path + str(nums) + '.tif'
                    self.write_tiff(filename, tmp_arr, split_size, split_size, 1)
                    nums += 1

                else:
                    split_arr = img.ReadAsArray(xoff=j * split_size, yoff=i * split_size, xsize=split_size,
                                                ysize=split_size)
                    filename = split_label_path + str(nums) + '.tif'
                    self.write_tiff(filename, split_arr, split_size, split_size, 1)
                    nums += 1

    def _bytes_features(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def make_train_file_tif_to_tfrecords(self, file15_path, file17_path, label_path, tf_file_path):
        write = tf.python_io.TFRecordWriter(tf_file_path)
        for i in range(1224):
            file15_name = file15_path + str(i) + '.tif'
            file17_name = file17_path + str(i) + '.tif'
            im15 = gdal.Open(file15_name)
            im17 = gdal.Open(file17_name)
            im15_arr = im15.ReadAsArray()
            im17_arr = im17.ReadAsArray()
            im_arr = np.concatenate((im15_arr, im17_arr), axis=0)
            im_arr = np.array(im_arr, dtype='float32')
            im_arr = np.transpose(im_arr, [1, 2, 0])

            label_path_name = label_path + str(i) + '.tif'
            label = gdal.Open(label_path_name)
            label_arr = label.ReadAsArray()
            for i in range(224):
                for j in range(224):
                    if label_arr[i][j] <= 0:
                        label_arr[i][j] = 0
                    else:
                        label_arr[i][j] = 1
            label_arr = np.array(label_arr, dtype='int32')

            img_raw = im_arr.tostring()
            label_str = label_arr.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': self._bytes_features(img_raw),
                'label': self._bytes_features(label_str)
            }))

            write.write(example.SerializeToString())
        write.close()

    def make_test_file_tif_to_tfrecords(self, file15_path, file17_path, tf_file_path):
        write = tf.python_io.TFRecordWriter(tf_file_path)
        for i in range(1224):
            file15_name = file15_path + str(i) + '.tif'
            file17_name = file17_path + str(i) + '.tif'
            im15 = gdal.Open(file15_name)
            im17 = gdal.Open(file17_name)
            im15_arr = im15.ReadAsArray()
            im17_arr = im17.ReadAsArray()
            im_arr = np.concatenate((im15_arr, im17_arr), axis=0)
            im_arr = np.array(im_arr, dtype='float32')
            im_arr = np.transpose(im_arr, [1, 2, 0])

            img_raw = im_arr.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': self._bytes_features(img_raw),
            }))

            write.write(example.SerializeToString())
        write.close()

    def read_bacth_multifile(self, filein_path, bacth_size):
        reader = tf.TFRecordReader()
        files = tf.train.match_filenames_once(filein_path)

        filename_queue = tf.train.string_input_producer(files)
        _, queue_batch = reader.read_up_to(filename_queue, bacth_size)
        features = tf.parse_example(queue_batch, features={
            'img_raw': tf.FixedLenFeature([], tf.string),
        })

        imgs = tf.decode_raw(features['img_raw'], tf.float32)

        images_bacth = tf.reshape(imgs, (bacth_size, 224, 224, 8))

        return files, images_bacth

    def read_batch(self, filename, batch_size):
        # filename= '/home/zb/split_hhh/train15.tfrecords'
        file_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, queue_batch = reader.read_up_to(file_queue, batch_size)
        feature = tf.parse_example(queue_batch, features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })
        imgs = tf.decode_raw(feature['img_raw'], tf.float32)
        labels = tf.decode_raw(feature['label'], tf.int32)

        img_batch = tf.reshape(imgs, (batch_size, 224, 224, 8))
        label_batch = tf.reshape(labels, (batch_size, 224, 224))

        return img_batch, label_batch


class SegNet():
    def __init__(self, num_epochs, learning_rate, learning_rate_decay, moving_average_decay):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.moving_average_decay = moving_average_decay

    # batch normalization
    def batch_normalization(self, input, isTraining, name):
        return tf.cond(isTraining,
                       lambda: tf.contrib.layers.batch_norm(input, is_training=isTraining, scope=name + '_bn'),
                       lambda: tf.contrib.layers.batch_norm(input, is_training=isTraining, scope=name + '_bn',
                                                            reuse=True))  # reuse=True

    def conv_op(self, input, kernel_shape, stride, isTraining, activation=True, name=None):
        depth = kernel_shape[3]
        with tf.variable_scope('conv' + name) as scope:

            kernel = tf.get_variable('conv-kernel', shape=kernel_shape,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())  # attention
            '''
            kernel = tf.get_variable('conv-kernel', shape=kernel_shape,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))  # attention
            '''

            biases = tf.get_variable('conv-biases', shape=[depth],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input, kernel, strides=[1, stride, stride, 1], padding='SAME', name=scope.name)
            layer = tf.nn.bias_add(conv, biases)

        if activation is True:
            relu_val = tf.nn.relu(self.batch_normalization(layer, isTraining, scope.name))
        else:
            relu_val = self.batch_normalization(layer, isTraining, scope.name)
        return relu_val

    def deconv_op(self, input, kernel_shape, out_shape, stride, isTraining, activation=True, name=None):
        # attention kernel_shape's [2] [3] is changed, cause it's deconvolution
        depth = kernel_shape[2]
        with tf.variable_scope('deconv' + name) as scope:
            kernel = tf.get_variable('deconv-kernel', shape=kernel_shape,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())  # attention
            '''
            kernel = tf.get_variable('deconv-kernel', shape=kernel_shape,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))  # attention
            '''

            biases = tf.get_variable('deconv-biases', shape=[depth],
                                     initializer=tf.constant_initializer(0.0))

            deconv = tf.nn.conv2d_transpose(input, kernel, out_shape, strides=[1, stride, stride, 1],
                                            padding='SAME',
                                            name=scope.name)
            layer = tf.nn.bias_add(deconv, biases)
        if activation is True:
            relu_val = tf.nn.relu(self.batch_normalization(layer, isTraining, scope.name))
        else:
            relu_val = self.batch_normalization(layer, isTraining, scope.name)
        return relu_val

    def max_pool_with_argmax_op(self, input, stride, name):
        # tf.nn.max_pool_with_argmax, only GPU-supported, tf.nn.max_pool_with_argmax_and_mask seem to high version support
        return tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME',
                                          name=name)

    def unpool(self, net, mask, stride, name):
        assert mask is not None
        with tf.name_scope(name):
            ksize = [1, stride, stride, 1]
            input_shape = net.get_shape().as_list()
            #  calculation new shape
            output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
            feature_range = tf.range(output_shape[3], dtype=tf.int64)
            f = one_like_mask * feature_range
            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(net)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(net, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def inference(self, images, batch_size, isTraining):
        # Local Response Normalization(局部响应归一化), 这里参数位经验值，论文中推荐的参数
        # 试一下，后期使用全局归一化的方法
        img_lrn = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='img_lrn')

        # encode
        conv1_1 = self.conv_op(img_lrn, [3, 3, img_lrn.get_shape().as_list()[3], 64], stride=1, isTraining=isTraining,
                               name='1_1')  # 224*224*8  -> 224*224*64
        pool1, arg1 = self.max_pool_with_argmax_op(conv1_1, stride=2, name='pool1')  # 112*112*64

        conv2_1 = self.conv_op(pool1, [3, 3, 64, 128], stride=1, isTraining=isTraining, name='2_1')  # 112*112*128
        pool2, arg2 = self.max_pool_with_argmax_op(conv2_1, stride=2, name='pool2')  # 56*56*128

        conv3_1 = self.conv_op(pool2, [3, 3, 128, 256], stride=1, isTraining=isTraining, name='3_1')  # 56*56*256
        pool3, arg3 = self.max_pool_with_argmax_op(conv3_1, stride=2, name='pool3')  # 28*28*256

        conv4_1 = self.conv_op(pool3, [3, 3, 256, 512], stride=1, isTraining=isTraining, name='4_1')  # 28*28*512
        pool4, arg4 = self.max_pool_with_argmax_op(conv4_1, stride=2, name='pool4')  # 14*14*512

        # conv5_1 = self.conv_op(pool4, [3, 3, 512, 512], stride=1, isTraining=isTraining, name='5_1')  # 14*14*512



        # decode
        unsampling1 = self.unpool(pool4, arg4, stride=2, name='unsampling1')  # 28*28*512
        deconv1 = self.deconv_op(unsampling1, [3, 3, 256, 512], [batch_size, 28, 28, 256], stride=1,
                                 isTraining=isTraining, name='un1')  # 28*28*256
        conv_decode1_1 = self.conv_op(deconv1, [3, 3, 256, 256], stride=1, isTraining=isTraining, activation=False,
                                      name='de1_1')  # 28*28*256

        unsampling2 = self.unpool(conv_decode1_1, arg3, stride=2, name='unsampling2')  # 56*56*256
        deconv2 = self.deconv_op(unsampling2, [3, 3, 128, 256], [batch_size, 56, 56, 128], stride=1,
                                 isTraining=isTraining, name='un2')  # 56*56*128
        conv_decode2_1 = self.conv_op(deconv2, [3, 3, 128, 128], stride=1, isTraining=isTraining, activation=False,
                                      name='de2_1')  # 56*56*128

        unsampling3 = self.unpool(conv_decode2_1, arg2, stride=2, name='unsampling3')  # 112*112*128
        deconv3_1 = self.deconv_op(unsampling3, [3, 3, 64, 128], [batch_size, 112, 112, 64], stride=1,
                                   isTraining=isTraining, name='un3')  # 112*112*128
        conv_decode3_1 = self.conv_op(deconv3_1, [3, 3, 64, 64], stride=1, isTraining=isTraining, activation=False,
                                      name='de3_1')  # 112*112*64

        unsampling4 = self.unpool(conv_decode3_1, arg1, stride=2, name='unsampling4')  # 224*224*64
        deconv4 = self.deconv_op(unsampling4, [3, 3, 64, 64], [batch_size, 224, 224, 64], stride=1,
                                 isTraining=isTraining, name='un4')  # 112*112*64
        conv_decode4_1 = self.conv_op(deconv4, [3, 3, 64, 64], stride=1, isTraining=isTraining, activation=False,
                                      name='de4_1')  # 28*28*512

        # classifier
        with tf.variable_scope('conv-classifier'):
            kernel = tf.get_variable('kernel-classifier', shape=[3, 3, 64, 2],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())  # attention
            '''
            kernel = tf.get_variable('conv-kernel', shape=kernel_shape,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))  # attention
            '''

            biases = tf.get_variable('biases-classifier', shape=2,
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(conv_decode4_1, kernel, strides=[1, 1, 1, 1], padding='SAME', name='conv-classifier')
            classifier = tf.nn.bias_add(conv, biases)

        # classifier = self.conv_op(conv_decode4_1, [3, 3, 64, 2], stride=1, isTraining=isTraining, name='end')
        prediction = self.prediction_result(classifier)

        return prediction

    def prediction_result(self, conv_result):
        # label = tf.cast(label, tf.int32)
        min_val = tf.constant(value=1.0e-10)
        conv_result = tf.reshape(conv_result, (-1, 2))  # 2 represent the class is 2
        conv_result = conv_result + min_val
        softmax = tf.nn.softmax(conv_result)  # (batch_size, height, width, channels)

        return softmax

    def cal_loss(self, prediction, labels):
        class_weight = np.array([1.5, 98.5], dtype='float32')
        min_val = tf.constant(value=1.0e-10)
        with tf.name_scope('loss'):
            labels = tf.reshape(tf.one_hot(labels, depth=2), (-1, 2))

            cross_entropy = -tf.reduce_sum(tf.multiply((labels * tf.log(prediction + min_val)), class_weight),
                                           axis=1)  # [1]
            # cross_entropy = -tf.reduce_sum(tf.multiply(labels, tf.log(prediction + min_val)), axis=1)

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)
            loss = tf.add_n(tf.get_collection('losses'), name='total_losses')
        return loss

    def train(self, batch_size, train_filename, model_path):
        impro = improcess()
        img_batch, label_batch = impro.read_batch(train_filename, batch_size)
        isTrain = tf.constant(True, dtype=tf.bool)

        prediction = self.inference(img_batch, batch_size, isTrain)

        label = tf.reshape(tf.one_hot(label_batch, depth=2), (-1, 2))
        acc_val = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(label, 1))

        '''
        labels = tf.reshape(label_batch, (-1, 1))
        acc_val = tf.equal(tf.arg_max(prediction, 1), tf.cast(labels, dtype=tf.int64), name='acc')
        '''
        accuracy = tf.reduce_mean(tf.cast(acc_val, dtype=tf.float32), name='acc')

        losses = self.cal_loss(prediction, label_batch)

        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   1224 / batch_size,
                                                   self.learning_rate_decay)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)
        # train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            num = 1224 / batch_size

            for i in range(self.num_epochs):
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print 'Epoch %s/%s ' % (i+1, self.num_epochs)
                for j in range(num):
                    _, setp, acc, loss = sess.run([train_op, global_step, accuracy, losses])
                    if j == 102:
                        print '102/1224   >..................................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 204:
                        print '204/1224   --->...............................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 306:
                        print '306/1224   ------>............................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 408:
                        print '408/1224   --------->.........................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 510:
                        print '510/1224   ------------>......................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 612:
                        print '612/1224   --------------->...................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 714:
                        print '714/1224   ------------------>................losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 816:
                        print '816/1224   --------------------->.............losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 918:
                        print '918/1224   ------------------------>..........losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 1020:
                        print '1020/1224   --------------------------->......losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 1122:
                        print '1122/1224   ------------------------------>...losses: %s    accuracy: %s' % (loss, acc)
                    elif j == 1224:
                        print '1224/1224   ------------------------------>...losses: %s    accuracy: %s' % (loss, acc)

                if (i % 20 == 0) & (i != 0):
                    saver.save(sess, model_path+str(i))



            saver.save(sess, model_path)

            coord.request_stop()
            coord.join(threads)

    def prediction(self, batch_size, model_path, test_tf_file_path, pre_file_out_path=None):
        # 1564 = 4 * 23 * 17
        isTrain = tf.constant(True, dtype=tf.bool)
        # isTrain = tf.constant(False, dtype=tf.bool)
        impro = improcess()
        # files, images = impro.read_bacth_multifile(self.predict_path, batch_size)
        # files, images = impro.read_bacth_multifile(test_tf_file_path, batch_size)

        img_batch, label_batch = impro.read_batch(test_tf_file_path, batch_size)

        prediction = self.inference(img_batch, batch_size, isTrain)
        result = tf.arg_max(prediction, 1)

        label = tf.reshape(tf.one_hot(label_batch, depth=2), (-1, 2))
        acc_val = tf.equal(result, tf.arg_max(label, 1))

        accuracy = tf.reduce_mean(tf.cast(acc_val, dtype=tf.float32), name='acc')

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        out_arr = np.zeros((4032, 15232))
        tmp_rows = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver.restore(sess, model_path)

            for i in range(1224):
                pre_result, acc = sess.run([result, accuracy])
                print '%s batch_size---->accuracy: %s' % (i, acc)
                tmp_val = pre_result.reshape(224, 224)
                if (i % 68 == 0) & (i != 0):
                    tmp_rows += 1
                out_arr[tmp_rows * 224: (tmp_rows + 1) * 224, (i%68)*224: (i%68+1)*224] += tmp_val

            coord.request_stop()
            coord.join(threads)

        out_arr = np.array(out_arr, dtype='int32')
        out_string = out_arr.tostring()

        tf.gfile.FastGFile(pre_file_out_path, 'wb').write(out_string)

    def prediction_no_label_compare(self, batch_size, model_path, test_tf_file_path, pre_file_out_path=None):
        # 1564 = 4 * 23 * 17
        isTrain = tf.constant(True, dtype=tf.bool)
        impro = improcess()
        # files, images = impro.read_bacth_multifile(self.predict_path, batch_size)
        files, images = impro.read_bacth_multifile(test_tf_file_path, batch_size)

        #img_batch, label_batch = impro.read_batch(test_tf_file_path, batch_size)

        prediction = self.inference(images, batch_size, isTrain)
        result = tf.arg_max(prediction, 1)

        #label = tf.reshape(tf.one_hot(label_batch, depth=2), (-1, 2))
        #acc_val = tf.equal(result, tf.arg_max(label, 1))

        #accuracy = tf.reduce_mean(tf.cast(acc_val, dtype=tf.float32), name='acc')

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        out_arr = np.zeros((4032, 15232))
        tmp_rows = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver.restore(sess, model_path)

            for i in range(1224):
                #pre_result, acc = sess.run([result, accuracy])
                pre_result = sess.run(result)
                #print '%s batch_size---->accuracy: %s' % (i, acc)
                tmp_val = pre_result.reshape(224, 224)
                if (i % 68 == 0) & (i != 0):
                    tmp_rows += 1
                out_arr[tmp_rows * 224: (tmp_rows + 1) * 224, (i%68)*224: (i%68+1)*224] += tmp_val

            coord.request_stop()
            coord.join(threads)

        out_arr = np.array(out_arr, dtype='int32')
        out_string = out_arr.tostring()

        tf.gfile.FastGFile(pre_file_out_path, 'wb').write(out_string)




    def generate_tif_file_from_txt(self, txt_file_in, tif_file_out):
        txt_read = open(txt_file_in)
        txt_data = txt_read.read()
        im_arr = np.fromstring(txt_data, dtype='int32')
        im_arr = im_arr.reshape(4032, 15232)  # recurrent split large image
        im_arr1 = im_arr[0: 4000, 0: 15106]  # recurrent raw image

        file_compare = '/media/zb/ml/raw_data/1/preliminary1_raw_data/tinysample.tif'
        im_tiny = gdal.Open(file_compare)
        impro = improcess()
        print im_tiny.GetProjection()

        impro.writeTiff(im_arr1, 15106, 4000, 1, tif_file_out, [0, 12, 0, 0, 0, -12], im_tiny.GetProjection())


    def prediction_new_local_run(self, batch_size, model_path, test_tf_file_path, pre_file_out_path=None):
        # 1564 = 4 * 23 * 17
        isTrain = tf.constant(True, dtype=tf.bool)
        impro = improcess()
        # files, images = impro.read_bacth_multifile(self.predict_path, batch_size)
        files, images = impro.read_bacth_multifile(test_tf_file_path, batch_size)

        #img_batch, label_batch = impro.read_batch(test_tf_file_path, batch_size)

        prediction = self.inference(images, batch_size, isTrain)
        result = tf.arg_max(prediction, 1)

        #label = tf.reshape(tf.one_hot(label_batch, depth=2), (-1, 2))
        #acc_val = tf.equal(result, tf.arg_max(label, 1))

        #accuracy = tf.reduce_mean(tf.cast(acc_val, dtype=tf.float32), name='acc')

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        out_arr = np.zeros((4032, 15232))
        tmp_rows = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver.restore(sess, model_path)

            for i in range(1224):
                #pre_result, acc = sess.run([result, accuracy])
                pre_result = sess.run(result)
                #print '%s batch_size---->accuracy: %s' % (i, acc)
                tmp_val = pre_result.reshape(224, 224)
                if (i % 68 == 0) & (i != 0):
                    tmp_rows += 1
                out_arr[tmp_rows * 224: (tmp_rows + 1) * 224, (i%68)*224: (i%68+1)*224] += tmp_val

            coord.request_stop()
            coord.join(threads)

        out_arr = np.array(out_arr, dtype='int32')
        im_arr = out_arr[0: 4000, 0: 15106]

        impro.write_tiff(pre_file_out_path, im_arr, 15106, 4000, 1)



    def operation(self):
        train_file_224 = 'oss://remotecom/train_newest/train_file/rematch_train_newest.tfrecords'  ############
        model_path = 'oss://remotecom/train_newest/model/model_newest.ckpt'
        # model_path = 'oss://remotecom/train_224_new/model001/model_224_001.ckpt'


        net = SegNet(200, 0.0015, 0.99, 0.9999)
        # operation 1: train operation  in pai
        '''
        train_batch_size = 1  # 1474 = 67 * 11 * 2    # 1139=17*67
        net.train(train_batch_size, train_file_224, model_path)
        '''

        # operation 2: predit result save in txt file in pai
        test_tf_file_path = 'oss://remotecom/rematch_train_224/test_file/rematch_test.tfrecords'
        prediction_out_path = 'oss://remotecom/train_newest/pre_txt/pre_newest.txt'

        # test operation
        '''
        test_batch_size = 1
        #net.prediction(test_batch_size, model_path, test_tf_file_path, prediction_out_path)
        net.prediction_no_label_compare(test_batch_size, model_path, test_tf_file_path, prediction_out_path)
        '''

        # operation 3: generate tif file     local operation
        # '''
        txt_path = '/home/zb/下载/pre_newest.txt'
        tif_file_out = '/media/zb/ml/pre_newest.tif'
        net.generate_tif_file_from_txt(txt_path, tif_file_out)
        # '''


if __name__ == '__main__':
    # prediction operation
    # 补充相应的地址信息即可通过模型预测结果
    impro = improcess()
    im15_path = ''    # 2015tif存放的位置
    im17_path = ''    # 2017tif存放的位置
    im15_split_save_path = ''   # 2015tif切割后存放的位置
    im17_split_save_path = ''   # 2017tif切割后存放的位置
    test_tf_save_path = ''   # 预测数据转为tfrecords后存放的位置
    model_path = ''   # 模型存放的位置
    result_path = '' # 输出tif文件的位置

    impro.adaptive_split_img(im15_path, im15_split_save_path, 224)
    impro.adaptive_split_img(im17_path, im17_split_save_path, 224)
    impro.make_test_file_tif_to_tfrecords(im15_split_save_path, im17_split_save_path, test_tf_save_path)

    net = SegNet(100, 0.0015, 0.99, 0.9999)
    net.prediction_new_local_run(1, model_path, test_tf_save_path, result_path)















