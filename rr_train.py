import tensorflow as tf
import numpy as np
import com.handysoft.py3.heart_failure_detection.rrdata as rrdata

FLAGS = tf.app.flags.FLAGS
FLAGS.image_size = 28
FLAGS.image_color = 1
FLAGS.batch_size = 100
FLAGS.learning_rate = 0.001
FLAGS.num_classes = 4
FLAGS.training_epochs = 20
FLAGS.log_dir = './rr_log/'
FLAGS.ckpt_dir = './rr_ckpt/'


class CNNModel:

    def __init__(self, sess, name):
        # session
        self.sess = sess
        self.name = name

        # place holders
        self.x = tf.placeholder(tf.float32, [None, FLAGS.image_size * FLAGS.image_size])
        self.x_img = tf.reshape(self.x, [-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
        self.y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # build model
        self._build_model()

    def _build_model(self):

        with tf.name_scope('conv_1'):
            conv1 = tf.layers.conv2d(inputs=self.x_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)

        with tf.name_scope('conv_2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)

        with tf.name_scope('conv_3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)

        with tf.name_scope('fc_1'):
            flat = tf.reshape(pool3, [-1, 128 * 4 * 4])
            dense = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout = tf.nn.dropout(dense, self.keep_prob)

        with tf.name_scope('final_out'):
            self.logits = tf.layers.dense(inputs=dropout, units=4)

        # define cost/loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        tf.summary.scalar('cost', self.cost)

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost)

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

    def get_accuracy(self, x_test, y_test, keep_prob):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.x: x_data, self.y: y_data, self.keep_prob: keep_prob})


def main():
    # get training data
    tf.set_random_seed(777)  # reproducibility
    raw_data, label_data = rrdata.get_data()
    len_data = len(raw_data)

    # build model
    sess = tf.Session()
    models = []
    num_models = 3
    for m in range(num_models):
        models.append(CNNModel(sess, "model" + str(m)))

    # initialize
    sess.run(tf.global_variables_initializer())
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # train my model
    print('Learning started.')
    print('Training ', len_data, ' files')
    print('Testing ', len_data, ' files')

    for epoch in range(FLAGS.training_epochs):
        avg_cost_list = np.zeros(len(models))

        for m_idx, m in enumerate(models):
            c, _ = m.train(raw_data, label_data, 0.7)
            avg_cost_list[m_idx] += c / len_data

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

        # accuracy & log writer
        feed_dict = {}
        for m_idx, m in enumerate(models):
            print('Model ', m_idx, ' Accuracy:', m.get_accuracy(raw_data, label_data, 1))
            feed_dict[m.x] = raw_data
            feed_dict[m.y] = label_data
            feed_dict[m.keep_prob] = 1

        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, epoch)
        summary_writer.flush()

    print('Learning Finished!')

    # save model
    saver = tf.train.Saver()
    save_path = saver.save(sess, FLAGS.ckpt_dir + "rr_train.ckpt")
    print("Model saved in file: ", save_path)


main()
