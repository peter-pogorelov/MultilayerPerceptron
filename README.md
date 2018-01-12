# MultilayerPerceptron
Multilayer perceptron C++ implementation without external dependencies (except STL)

####----------------------------------------------------------------------
### test data from Andrew Ng course (flower dataset):
### neural network: 4x1
### activations: tanh x sigmoid
#### iteration: 0, cost: 138.852516
#### iteration: 1000, cost: 56.552355
#### iteration: 2000, cost: 50.345933
#### iteration: 3000, cost: 48.804189
#### iteration: 4000, cost: 47.756347
#### iteration: 4900, cost: 46.872858'
#### accuracy: 0.920000

####----------------------------------------------------------------------
### breast cancer sklearn dataset
### neural network: 10x5x2x1 neurons
### activations: tanh x relu x tanh x sigmoid
### + LR decay
### + neurons L2 regularization
### + bengio initialization
#### epoch: 0, cost: 138.629436, lr: 1.000000 
#### epoch: 1000, cost: 111.128019, lr: 0.367695 
#### epoch: 2000, cost: 118.021834, lr: 0.135200 
#### epoch: 3000, cost: 98.248265, lr: 0.049712 
#### epoch: 4000, cost: 99.386037, lr: 0.018279 
#### epoch: 4900, cost: 101.369613, lr: 0.007428 
accuracy: 0.700000
