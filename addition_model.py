    
import tensorflow as tf
from model import *    


if __name__ == '__main__':
        
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    model = Model(filename_list = ['training_data/add_op_data.tfrecords'],
        training_size = 2 ** 12,
        batch_size = 128,
        seed = 0,
        initial_learning_rate = .01,
        dropout_rate = 0.0625,
        regularization_scale = .25,
        layer_size = 96,
        layer_depth = 2,
        hidden_layer_activation = tf.nn.leaky_relu,
        sess = sess
        )
    
    
    sess.run(tf.global_variables_initializer())

    for i in range(50000):        
        if i % 25 == 0:
            metrics = model.metrics()
            learning_rate = sess.run(model.learning_rate)
            print('Epoch ' + str(i) + ' - Addition Model')
            print('Learning rate: ' + str(learning_rate))
            print(metrics)
            print('\n')
            
            print('Example Probabilities:\t' + str(sess.run(model.probabilities, feed_dict = {
                model.x: [[0, 0, 0, 0, 1, 1, 0, 1]], model.y: [[0, 1, 0, 0, 0, 1, 0, 1]], model.training_phase: False})))
            
            print('Example Prediction:\t' + str(model.make_predictions([[0., 0., 0., 0., 1., 1., 0., 1.]], 
                                                          [[0., 1., 0., 0., 0., 1., 0., 1.]])))
            print('Expected Result:\t[[ 0., 1., 0., 1., 0., 0., 1., 0.]]\n')

                          
        model.train_for_one_epoch()
        



