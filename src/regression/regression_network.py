import tensorflow as tf
import numpy as np
from tqdm import trange, tqdm

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class RegressionNetwork:
    '''
    Class holding regression network

    '''
    def __init__(self, images, target, save_loc, embed_size=64, epochs=10, num_batches=15):
        '''
            images: LxHxDxN vector of images
            target: Nx1 vector of target regression values
            embed_size: size of image embedding
            epochs: number of training epochs
            batch_size: batch size
        '''

        self.embed_size=embed_size
        self.epochs = epochs
        self.num_batches = num_batches
        self.save_loc = save_loc #TODO: optional loading

        #split data into training + test
        self.all_images = images.transpose((3, 0, 1, 2)).astype(np.float32)
        self.all_targets = target

        self.sess = tf.Session()
        self.split_data()

        #placeholder variables
        self.X = tf.placeholder(tf.float32, shape=[None, self.train_imgs.shape[1], self.train_imgs.shape[2], self.train_imgs.shape[3]])
        self.Y = tf.placeholder(tf.float32, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)

        print("initializing network")
        self.feats, self.output_prediction = self.build_network()

        with tf.name_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.Y, self.output_prediction)

        tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        #add FileWriter for TensorBoard
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.save_loc + '/train', self.sess.graph)


    def split_data(self, num_train=0.9):
        '''
        Splits data into random training and test set
        '''

        num_data = self.all_images.shape[0]
        inds = np.arange(0, num_data)

        np.random.shuffle(inds)

        train_ind = int(num_data*num_train)

        train_inds = inds[:train_ind]
        test_inds = inds[train_ind:]

        self.train_imgs = self.all_images[train_inds,:,:,:]

        self.train_targets = self.all_targets[train_inds]

        self.test_imgs = self.all_images[test_inds,:,:,:]

        self.test_targets = self.all_images[test_inds]


    def build_network(self):
        '''
        Constructs regression CNN.
        '''

        #convolutional layers
        with tf.name_scope("conv1"):
            c1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[25,25], padding="same", activation=tf.nn.relu, name="conv1")
            mp1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[10, 10], strides=5, name="max_pool1")

        with tf.name_scope("conv2"):
            c2 = tf.layers.conv2d(inputs=mp1, filters=32, kernel_size=[10, 10], padding="same", activation=tf.nn.relu, name="conv2")
            mp2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[4,4], strides=2, name="max_pool2")

        with tf.name_scope("conv3"):
            c3 = tf.layers.conv2d(inputs=mp2, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="conv3")
            mp3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=2, name="max_pool3")

        #dense layers w/dropout
        mp3_flat = tf.reshape(mp3, [-1, mp3.get_shape()[1]*mp3.get_shape()[2]*mp3.get_shape()[3]])


        with tf.name_scope("dense1"):
            dense1 = tf.layers.dense(inputs=mp3_flat, units=1024, activation=tf.nn.relu, name="dense1")
            dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob)

        #feature embedding layer
        with tf.name_scope("feats"):
            feats = tf.layers.dense(inputs=dropout, units=self.embed_size, activation=tf.nn.relu, name="feats")

        #predicts single regression output
        with tf.name_scope("output"):
            output = tf.layers.dense(inputs=feats, units=1, activation=tf.nn.relu, name="output")
            #'flatten' to avoid shape issues
            output_prediction = tf.reshape(output, [-1])

        return feats, output_prediction


    def train(self, epochs=10):
        '''
        Trains the network.
        '''

        batch_indices = np.array_split(np.arange(self.train_imgs.shape[0]), self.num_batches)

        nbatches = len(batch_indices)

        #TODO: progress bar
        print("training! ")
        epochs = trange(self.epochs)
        for e in epochs:
            epochs.set_description("Epoch {}".format(e))
            sum_loss = 0.0
            batches = tqdm(batch_indices)
            for i, batch in enumerate(batches) :
                batches.set_description("Batch {}".format(i))
                loss, summary, _ = self.sess.run([self.loss,  self.merged_summary, self.train_op,], feed_dict={self.X:self.train_imgs[batch,:,:,:],
                                                                                self.Y:self.train_targets[batch],
                                                                                self.keep_prob:0.4})
                batches.set_postfix(loss=loss)
                self.train_writer.add_summary(summary, e*i)
                sum_loss += loss
            epochs.set_postfix(avg_loss=sum_loss/nbatches)

        print("training complete, saving")
        self.saver.save(sess, self.save_loc)



    def test(self):
        '''
        Tests on test data
        '''

    def eval(self, img):
        '''
        Performs regression on single image.
        '''
