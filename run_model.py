import network
from network import Network
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import json

# loads the optimized parameter values
with open("params.json", "r") as f:
    data = json.load(f)

# creates QNN objects and loads parameters into it
net = Network(2, 2)
net.load_params(data)

# records output from QNN when the [0, 0] state is passed
# based on the training data the most frequent result should be [1, 0]
counts = net.pass_state([0, 0])
plot_histogram(counts, title="Test input-[0, 0]")
print(f"Result: {network.find_key(counts)}")

# shows the circuit
net.show()
