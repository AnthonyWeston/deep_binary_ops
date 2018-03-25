import tensorflow as tf
from data import Data

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
        self.learning_rate_global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step = 
            self.learning_rate_global_step, decay_steps = 100 * self.training_size // self.batch_size, 
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
        
    def evaluate_loss(self, dataset):
        feature_dict = sess.run(model.data.get_training_dataset_as_tensor_dict())
        
        print(sess.run(model.loss, feed_dict = {model.x: feature_dict['x'],
                                            model.y: feature_dict['y'],
                                            model.z: feature_dict['z'],
                                            model.training_phase: False}))

        
            
    def _create_hidden_layer(self, inputs):
        dense_layer = tf.layers.dense(inputs, self.layer_size, activation = None, 
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.regularization_scale), 
            activity_regularizer = tf.contrib.layers.l2_regularizer(self.regularization_scale),
            kernel_initializer = tf.initializers.random_normal(self.seed))
        
        batch_norm_layer = tf.layers.batch_normalization(dense_layer, training = self.training_phase)
        
        dropout_layer = tf.layers.dropout(batch_norm_layer, self.dropout_rate, 
            training = self.training_phase)
        
        output = self.hidden_layer_activation(dropout_layer)
        
        return output
    
    def _build_hidden_layers(self, inputs):
        hidden_layers = [None for _ in range(self.layer_depth)]
        
        
        hidden_layers[0] = self._create_hidden_layer(inputs)
        
        for i in range(self.layer_depth - 1):
            hidden_layers[i + 1] = self._create_hidden_layer(hidden_layers[i])
            
        return hidden_layers
    
"""
For manual testing of the model
"""
    
if __name__ == '__main__':
        
    sess = tf.Session()

    model = Model(filename_list = ['training_data/add_op_data.tfrecords'],
        training_size = 63488,
        batch_size = 3,
        seed = 0,
        initial_learning_rate = .05,
        dropout_rate = .5,
        regularization_scale = .1,
        layer_size = 10,
        layer_depth = 3,
        hidden_layer_activation = tf.nn.tanh,
        sess = sess
        )

    init = tf.global_variables_initializer()
    sess.run(init)
    
    feature_dict = sess.run(model.data.get_training_dataset_as_tensor_dict())
    print(feature_dict)
        
    print(sess.run(model.loss, feed_dict = {model.x: feature_dict['x'],
                                            model.y: feature_dict['y'],
                                            model.z: feature_dict['z'],
                                            model.training_phase: True}))








