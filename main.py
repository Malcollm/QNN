import matplotlib.pyplot as plt
from network import Network
from optimizer import Optimizer
import json

# this file is used to train the model

# creates a QNN object
net = Network(2, 2)
# this the training data - [[input vectors], [output vectors]]
data = [[[0, 0], [1, 1]], [[1, 0], [0, 1]]]

# creates a optimizer object
optimizer = Optimizer(data, net)
# optimizes the QNN object
optimizer.optimize(40,0.5)

# saves the optimized parameter values
with open("params.json", "w") as f:
    json.dump(net.params, f)
