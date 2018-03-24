import tensorflow as tf

class Model:
    
    BITS_PER_NUMBER = 8
    
    def __init__(self, filename_list: list, training_size: int, batch_size: int, seed: int,
                initial_learning_rate: float, dropout_rate: int, regularization_scale: int,
                layer_size: int, layer_depth: int):
        self.filename_list = filename_list
        self.training_size = training_size
        self.batch_size = batch_size
        self.seed = seed      
        self.dropout_rate = dropout_rate
        self.regularization_scale = regularization_scale
        self.layer_size = layer_size
        self.layer_depth = layer_depth
        
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step = 
            self.learning_rate_global_step, decay_steps = 100 * self.training_size // self.batch_size, 
            decay_rate = 0.95, staircase = True)

        self.x = tf.placeholder(tf.float32, shape = [None, Model.BITS_PER_NUMBER], name = 'X_input')
        self.y = tf.placeholder(tf.float32, shape = [None, Model.BITS_PER_NUMBER], name = 'Y_input')
        
        self.input = tf.concat([self.x, self.y], 1, 'ConcatenatedInput')
        
        self.training_phase = tf.placeholder(tf.bool)
        
    
    
    def hidden_layer(self, inputs):
        self.hidden_layer_1 = tf.layers.dense(input, self.layer_size, activation = None, 
            kernel_regularizer = tf.contrib.layers.l2_regularizer(regularization_scale), 
            activity_regularizer = tf.contrib.layers.l2_regularizer(regularization_scale=0.1))
        
        self.batch_norm_layer_1 = tf.layers.batch_normalization(self.hidden_layer_1, training = self.training_phase)
        self.dropout_layer_1 = tf.layers.dropout(self.batch_norm_layer_1, dropout_rate, training = self.training_phase, name = 'Dropout1')
        #self.leaky_relu_1 = tf.nn.leaky_relu(self.dropout_layer_1, alpha = 0.05, name = 'LeakyRelu1')
        self.sigmoid_1 = tf.nn.sigmoid(self.dropout_layer_1, name = 'Sigmoid1')

