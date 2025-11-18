from network import Network
from optimizer import Optimizer
import json
from math import pi

# this file is used to train the model

# creates a QNN object with 2 qubits and 2 layers
net = Network(2, 2)

# this the training data - [[input vectors], [output vectors]]
# each component in the vectors represents a qubit
# the input data is the degrees rotated around the x-axis bloch sphere (0 -> |0> pi -> |1>)
# the output data is classical and can be 1 (always on) 0 (always off) or some value in the middle like
# 0.5 (50% 1 50% 0)

# This is example of a very small data set where output is just the opposite state as the input
data = [[[0, 0], [pi, pi], [pi, 0]], [[1, 1], [0, 0], [0, 1]]]

# creates a optimizer object
optimizer = Optimizer(data, net)
# optimizes the QNN object
# performs 30 reps of gradient decent with the gradient vectored scaled by a factor of 0.2
optimizer.optimize(30,0.2)

# saves the optimized parameter values
with open("params.json", "w") as f:
    json.dump(net.params, f)
