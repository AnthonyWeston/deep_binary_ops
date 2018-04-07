# Deep Learning - Binary Operations Simulator

## Purpose

To develop deep learning skills using Tensorflow and simple datasets

## Requirements

- Python 3.5 or higher
- TensorFlow (Numpy is used in this project, but is also installed when installing TensorFlow using pip

## The Datasets

The datasets consist of three sets of eight bits - two 8-bit unsigned binary integer inputs and one 8-bit output, which is the result of applying a binary operation to the two inputs. 

Three datasets are generated for applying unsigned binary addition, unsigned binary multiplication, and the xor operation using the following command:

```
python generate_training_data.py
```

## The Model

The model is a multilayer perceptron (a standard neural network, nothing fancy), with the number and properties the hidden layers specified by the user. A single Model class was created to represent the TensorFlow model, and three scripts were created with unique parameters to train a model on each dataset.

## Training the Models

To train the models, use the following commands:

```
python addition_model.py
python multiplication_model.py
python xor_model.py
```


