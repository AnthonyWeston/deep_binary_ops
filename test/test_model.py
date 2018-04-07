import os
from model import Model
import tensorflow as tf
from data import Data
import numpy as np

test_file_path = os.path.dirname(os.path.realpath(__file__))


class TestModel(tf.test.TestCase):
    
    test_filename_list = [os.path.join(test_file_path, 'test_data.tfrecords')]
    training_size = 10
    batch_size = 5
    seed = 1
    initial_learning_rate = .05
    dropout_rate = .1
    regularization_scale = 0.01
    layer_size = 24
    layer_depth = 3
    hidden_layer_activation = tf.nn.tanh
    
    @classmethod
    def setUpClass(cls):
        TestModel.shared_model = Model(filename_list = TestModel.test_filename_list,
            training_size = TestModel.training_size,
            batch_size = TestModel.batch_size,
            seed = TestModel.seed,
            initial_learning_rate = TestModel.initial_learning_rate,
            dropout_rate = TestModel.dropout_rate,
            regularization_scale = TestModel.regularization_scale,
            layer_size = TestModel.layer_size,
            layer_depth = TestModel.layer_depth,
            hidden_layer_activation = TestModel.hidden_layer_activation,
            sess = None)
        
    
    def create_training_model(self):
        sess = tf.Session()

        training_model = Model(filename_list = TestModel.test_filename_list,
            training_size = TestModel.training_size,
            batch_size = TestModel.batch_size,
            seed = TestModel.seed,
            initial_learning_rate = TestModel.initial_learning_rate,
            dropout_rate = TestModel.dropout_rate,
            regularization_scale = TestModel.regularization_scale,
            layer_size = TestModel.layer_size,
            layer_depth = TestModel.layer_depth,
            hidden_layer_activation = TestModel.hidden_layer_activation,
            sess = sess)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        return training_model 


    
    def test_the_model_is_created_with_a_list_of_data_filenames(self):
        subject = TestModel.shared_model
        
        expected_filename_list = TestModel.test_filename_list
        self.assertEqual(expected_filename_list, subject.filename_list) 
        
    def test_the_model_is_created_with_a_training_set_size(self):
        subject = TestModel.shared_model
        
        expected_training_size = TestModel.training_size
        self.assertEqual(expected_training_size, subject.training_size)
        
    def test_the_model_is_created_with_a_training_batch_size(self):
        subject = TestModel.shared_model
        
        expected_batch_size = TestModel.batch_size
        self.assertEqual(expected_batch_size, subject.batch_size)
        
    def test_the_model_is_created_with_a_random_seed(self):
        subject = TestModel.shared_model
        
        expected_seed = TestModel.seed
        self.assertEqual(expected_seed, subject.seed)
        
    def test_the_model_is_created_with_an_initial_learning_rate(self):
        subject = TestModel.shared_model
        
        expected_learning_rate = TestModel.initial_learning_rate
        self.assertEqual(expected_learning_rate, subject.initial_learning_rate)
        
    def test_the_model_is_created_with_a_dropout_rate(self):
        subject = TestModel.shared_model
        
        expected_dropout_rate = TestModel.dropout_rate
        self.assertEqual(expected_dropout_rate, subject.dropout_rate)
        
    def test_the_model_is_created_with_a_regularization_scale(self):
        subject = TestModel.shared_model
        
        expected_regularization_scale = TestModel.regularization_scale
        self.assertEqual(expected_regularization_scale, subject.regularization_scale)
        
    def test_the_model_has_an_exponentially_decaying_learning_rate(self):
        subject = self.create_training_model()
                
        expected_decay = TestModel.initial_learning_rate
                
        self.assertAlmostEqual(expected_decay, subject.sess.run(subject.learning_rate), delta = .00000001)
        
    def test_the_model_is_created_with_a_layer_size(self):
        subject = TestModel.shared_model
        
        expected_layer_size = TestModel.layer_size
        self.assertEqual(expected_layer_size, subject.layer_size)
        
    def test_the_model_is_created_with_a_layer_depth(self):
        subject = TestModel.shared_model
        
        expected_layer_depth = TestModel.layer_depth
        self.assertEqual(expected_layer_depth, subject.layer_depth)   
        
    def test_the_model_has_an_x_placeholder_with_8_bits(self):
        subject = TestModel.shared_model.x

        expected_bits = Model.BITS_PER_NUMBER
        self.assertEqual(expected_bits, subject.shape[1])
        
    def test_the_model_has_a_y_placeholder_with_8_bits(self):
        subject = TestModel.shared_model.y

        expected_bits = Model.BITS_PER_NUMBER
        self.assertEqual(expected_bits, subject.shape[1])
        
    def test_the_model_has_a_training_mode_placeholder(self):
        subject = TestModel.shared_model.training_phase
        
        expected_dtype = tf.bool
        self.assertEqual(expected_dtype, subject.dtype)
        
    def test_the_model_is_created_with_a_hidden_layer_activation_function(self):
        subject = TestModel.shared_model

        expected_activation_function = TestModel.hidden_layer_activation
        self.assertEqual(expected_activation_function, subject.hidden_layer_activation)
        
    def test_the_model_has_a_data_object(self):
        subject = TestModel.shared_model

        self.assertIsInstance(subject.data, Data)
        
    def test_when_the_model_is_trained_its_loss_decreases(self):
        subject = self.create_training_model()
        subject.sess.run(tf.global_variables_initializer())    
        
        initial_loss = subject.evaluate_loss('train')
        for _ in range(2):
            subject.train_for_one_epoch()
        final_loss = subject.evaluate_loss('train')
        
        self.assertLessEqual(final_loss, initial_loss)
        
    def test_when_the_model_is_trained_its_accuracy_increases(self):
        subject = self.create_training_model()
        
        initial_accuracy = subject.evaluate_accuracy('train')
        for _ in range(100):
            subject.train_for_one_epoch()
        final_accuracy = subject.evaluate_accuracy('train')
      
        
        self.assertGreaterEqual(final_accuracy, initial_accuracy)
        
    def test_the_training_variables_are_changed_after_one_epoch(self):
        model = self.create_training_model()
        sess = model.sess

        before_variables = sess.run(tf.trainable_variables())
        
        model.train_for_one_epoch()
        
        after_variables = sess.run(tf.trainable_variables())
        
        for b, a in zip(before_variables, after_variables):
            self.assertTrue(np.any(np.not_equal(b,a)))
            
    def test_the_training_variables_do_not_stop_training_by_the_100th_epoch(self):
        model = self.create_training_model()
        sess = model.sess

        for i in range(100):
            print(i)

            before_variables = sess.run(tf.trainable_variables())
            
            model.train_for_one_epoch()
            
            after_variables = sess.run(tf.trainable_variables())
            
            for b, a in zip(before_variables, after_variables):
                self.assertTrue(np.any(np.not_equal(b,a)))
        