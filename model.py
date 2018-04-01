import tensorflow as tf
from data import Data
from tensorflow.python import debug as tf_debug

class Model:
    
    BITS_PER_NUMBER = 8
    
    def __init__(self, filename_list: list, training_size: int, batch_size: int, seed: int,
                initial_learning_rate: float, dropout_rate: int, regularization_scale: int,
                layer_size: int, layer_depth: int, hidden_layer_activation, sess: tf.Session):
        self.filename_list = filename_list
        self.training_size = training_size
        self.batch_size = batch_size
        self.seed = seed      
        self.dropout_rate = dropout_rate
        self.regularization_scale = regularization_scale
        self.layer_size = layer_size
        self.layer_depth = layer_depth
        self.hidden_layer_activation = hidden_layer_activation
        
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_global_step = tf.Variable(0, trainable=False, name = 'LearningRateGlobalStep')
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step = 
            self.learning_rate_global_step, decay_steps = self.training_size // self.batch_size, 
            decay_rate = 0.95, staircase = True)

        self.x = tf.placeholder(tf.float32, shape = [None, Model.BITS_PER_NUMBER], name = 'X_input')
        self.y = tf.placeholder(tf.float32, shape = [None, Model.BITS_PER_NUMBER], name = 'Y_input')
        self.z = tf.placeholder(tf.float32, shape = [None, Model.BITS_PER_NUMBER], name = 'Labels')
        self.training_phase = tf.placeholder(tf.bool)
        
        inputs = tf.concat([self.x, self.y], 1, 'ConcatenatedInput')

        self.hidden_layers = self._build_hidden_layers(inputs)
        
        self.logits = tf.layers.dense(inputs = self.hidden_layers[-1], units = Model.BITS_PER_NUMBER,
            activation = self.hidden_layer_activation)
        self.probabilities = tf.sigmoid(self.logits)
        
        self.loss = tf.losses.sigmoid_cross_entropy(self.z, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, 
            name = "AdamOptimizer").minimize(self.loss, global_step = self.learning_rate_global_step)
            
        self.data = Data(filename_list = self.filename_list, training_size = self.training_size, 
            batch_size = self.batch_size)
        
        self.sess = sess
        
        self.predictions = tf.round(self.probabilities, name = 'Predictions')
        self.prediction_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.y)),
            name = 'Accuracy')
        
    def evaluate_loss(self, dataset_type: str):
        sess = self.sess
        
        if dataset_type == 'train':
            feature_dict = self.data.get_training_dataset_as_dict()
        elif dataset_type == 'test':
            feature_dict = self.data.get_test_dataset_as_dict()

        return sess.run(self.loss, feed_dict = {self.x: feature_dict['x'],
                                            self.y: feature_dict['y'],
                                            self.z: feature_dict['z'],
                                            self.training_phase: False})
        
    def evaluate_accuracy(self, dataset_type: str):
        sess = self.sess
        
        if dataset_type == 'train':
            feature_dict = self.data.get_training_dataset_as_dict()
        elif dataset_type == 'test':
            feature_dict = self.data.get_test_dataset_as_dict()
            
        return sess.run(self.prediction_accuracy, feed_dict = {self.x: feature_dict['x'],
                                            self.y: feature_dict['y'],
                                            self.z: feature_dict['z'],
                                            self.training_phase: False})

    def train_for_one_epoch(self):
        for _ in range(self.data.num_batches): 
            feature_dict = self.data.get_training_batch_as_dict()

            self.sess.run(self.optimizer, feed_dict = {self.x: feature_dict['x'],
                                            self.y: feature_dict['y'],
                                            self.z: feature_dict['z'],
                                            self.training_phase: True})

            
    def _create_hidden_layer(self, inputs):
        dense_layer = tf.layers.dense(inputs, self.layer_size, activation = None, 
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.regularization_scale), 
            activity_regularizer = tf.contrib.layers.l2_regularizer(self.regularization_scale),
            kernel_initializer = tf.initializers.random_normal(self.seed))
        
        #batch_norm_layer = tf.layers.batch_normalization(dense_layer, training = self.training_phase)
        
        #dropout_layer = tf.layers.dropout(batch_norm_layer, self.dropout_rate, 
        #    training = self.training_phase)
        dropout_layer = tf.layers.dropout(dense_layer, self.dropout_rate, 
            training = self.training_phase)
        
        output = self.hidden_layer_activation(dropout_layer)
        
        return output
    
    def _build_hidden_layers(self, inputs):
        hidden_layers = [None for _ in range(self.layer_depth)]
        
        hidden_layers[0] = self._create_hidden_layer(inputs)
        
        for i in range(self.layer_depth - 1):
            hidden_layers[i + 1] = self._create_hidden_layer(hidden_layers[i])
            
        return hidden_layers
    
    def metrics(self):
        return """Training loss:    {0}
Training accuracy:    {1}
Test loss:    {2}
Test accuracy    {3}""".format(self.evaluate_loss('train'), self.evaluate_accuracy('train'),
            self.evaluate_loss('test'), self.evaluate_accuracy('test'))
    
"""
For manual testing and evaluation of the model
"""
    
    
if __name__ == '__main__':
        
    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    model = Model(filename_list = ['training_data/add_op_data.tfrecords'],
        training_size = 2 ** 14 - 1024,
        batch_size = 256,
        seed = 0,
        initial_learning_rate = .001,
        dropout_rate = .125,
        regularization_scale = .1,
        layer_size = 24,
        layer_depth = 3,
        hidden_layer_activation = tf.nn.tanh,
        sess = sess
        )
    
    sess.run(tf.global_variables_initializer())

    for i in range(5):        
        if i % 1 == 0:
            metrics = model.metrics()
            learning_rate = sess.run(model.learning_rate)
            print('Epoch ' + str(i))
            print('Learning rate: ' + str(learning_rate))
            print(metrics)
            print('\n')
            
        model.train_for_one_epoch()
        



