import tensorflow as tf
import numpy as np

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

        self.loss = tf.losses.mean_squared_error(self.Y, self.output_prediction)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


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
        c1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[25,25], padding="same", activation=tf.nn.relu)

        mp1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[10, 10], strides=5)

        c2 = tf.layers.conv2d(inputs=mp1, filters=32, kernel_size=[10, 10], padding="same", activation=tf.nn.relu)

        mp2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[4,4], strides=2)

        c3 = tf.layers.conv2d(inputs=mp2, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        mp3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=2)

        #dense layers w/dropout
        mp3_flat = tf.reshape(mp3, [-1, mp3.get_shape()[1]*mp3.get_shape()[2]*mp3.get_shape()[3]])

        dense1 = tf.layers.dense(inputs=mp3_flat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob)

        #feature embedding layer
        feats = tf.layers.dense(inputs=dropout, units=self.embed_size, activation=tf.nn.relu)

        #predicts single regression output
        output = tf.layers.dense(inputs=feats, units=1, activation=tf.nn.relu)

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
        for e in range(self.epochs):
            sum_loss = 0.0
            for i, batch in enumerate(batch_indices):
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.X:self.train_imgs[batch,:,:,:],
                                                                                self.Y:self.train_targets[batch],
                                                                                self.keep_prob:0.4})
                sum_loss += loss
            print("Average loss for epoch {}".format(loss/nbatches))

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
