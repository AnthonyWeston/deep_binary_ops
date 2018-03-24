import tensorflow as tf

class Model:
    
    def __init__(self, filename_list: list, training_size: int, batch_size: int, seed: int,
                initial_learning_rate: float, dropout_rate: int, regularization_scale: int):
        self.filename_list = filename_list
        self.training_size = training_size
        self.batch_size = batch_size
        self.seed = seed
        
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step = 
            self.learning_rate_global_step, decay_steps = 100 * self.training_size // self.batch_size, 
            decay_rate = 0.95, staircase = True)
        
        self.dropout_rate = dropout_rate
        self.regularization_scale = regularization_scale
        