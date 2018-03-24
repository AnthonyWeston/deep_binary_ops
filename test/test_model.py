import os
from model import Model
import tensorflow as tf

test_file_path = os.path.dirname(os.path.realpath(__file__))


class TestModel(tf.test.TestCase):
    
    test_filename_list = [os.path.join(test_file_path, 'test_data.tfrecords')]
    training_size = 10
    batch_size = 2
    seed = 1
    initial_learning_rate = 0.0005
    dropout_rate = .1
    regularization_scale = 0.01
    
    @classmethod
    def setUpClass(cls):
        TestModel.sess = tf.Session()    
    
    def setUp(self):
        self.model = Model(filename_list = TestModel.test_filename_list,
            training_size = TestModel.training_size,
            batch_size = TestModel.batch_size,
            seed = TestModel.seed,
            initial_learning_rate = TestModel.initial_learning_rate,
            dropout_rate = TestModel.dropout_rate,
            regularization_scale = TestModel.regularization_scale)
    
    def test_the_model_is_created_with_a_list_of_data_filenames(self):
        subject = self.model
        
        expected_filename_list = TestModel.test_filename_list
        self.assertEqual(expected_filename_list, subject.filename_list) 
        
    def test_the_model_is_created_with_a_training_set_size(self):
        subject = self.model
        
        expected_training_size = TestModel.training_size
        self.assertEqual(expected_training_size, subject.training_size)
        
    def test_the_model_is_created_with_a_training_batch_size(self):
        subject = self.model
        
        expected_batch_size = TestModel.batch_size
        self.assertEqual(expected_batch_size, subject.batch_size)
        
    def test_the_model_is_created_with_a_random_seed(self):
        subject = self.model
        
        expected_seed = TestModel.seed
        self.assertEqual(expected_seed, subject.seed)
        
    def test_the_model_is_created_with_an_initial_learning_rate(self):
        subject = self.model
        
        expected_learning_rate = TestModel.initial_learning_rate
        self.assertEqual(expected_learning_rate, subject.initial_learning_rate)
        
    def test_the_model_is_created_with_a_dropout_rate(self):
        subject = self.model
        
        expected_dropout_rate = TestModel.dropout_rate
        self.assertEqual(expected_dropout_rate, subject.dropout_rate)
        
    def test_the_model_is_created_with_a_regularization_scale(self):
        subject = self.model
        
        expected_regularization_scale = TestModel.regularization_scale
        self.assertEqual(expected_regularization_scale, subject.regularization_scale)
        
    def test_the_model_has_an_exponentially_decaying_learning_rate(self):
        subject = self.model
        
        global_step = tf.Variable(0, trainable=False)
        
        expected_decay = tf.train.exponential_decay(TestModel.initial_learning_rate, global_step = 
            global_step, decay_steps = 100 * TestModel.training_size // TestModel.batch_size, 
            decay_rate = 0.95, staircase = True)
        
        TestModel.sess.run(tf.global_variables_initializer())
        
        self.assertEqual(TestModel.sess.run(expected_decay), TestModel.sess.run(subject.learning_rate))
        
        
        
        
        
        
        
        