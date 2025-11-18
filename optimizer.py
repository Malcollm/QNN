import math


class Optimizer:
    def __init__(self, data, net):
        self.net = net
        self.layers = net.get_shape()[0]
        self.q_bits = net.get_shape()[1]

        self.data = data
        self.start_error = 0
        self.end_error = 0

        # creates empty parameter list
        self.adj = []
        for i in range(self.layers):
            self.adj.append([])
            for j in range(3):
                self.adj[i].append([])
                for k in range(self.q_bits):
                    self.adj[i][j].append(0)

    def optimize_step(self, step):
        for layer in range(self.layers):
            for r in range(3):
                for n in range(self.q_bits):
                    self.start_error = 0
                    self.end_error = 0
                    # calculates the component of gradient vector of the error function for the given parameter
                    for data_point in range(len(self.data[0])):
                        self.net.set_param(layer, n, r, self.net.get_param(layer, n, r) - math.pi / 2)
                        self.net.pass_state(self.data[0][data_point])
                        self.start_error += self.net.get_error(self.data[1][data_point])

                        self.net.set_param(layer, n, r, self.net.get_param(layer, n, r) + math.pi)
                        self.net.pass_state(self.data[0][data_point])
                        self.end_error += self.net.get_error(self.data[1][data_point])

                        self.net.set_param(layer, n, r, self.net.get_param(layer, n, r) - math.pi/2)

                    self.adj[layer][r][n] = -step*(self.end_error - self.start_error) / 2

        # adjusts the parameters based on the gradient vector
        for layer in range(self.layers):
            for r in range(3):
                for n in range(self.q_bits):
                    self.net.set_param(layer, n, r, self.net.get_param(layer, n, r) + self.adj[layer][r][n])

        # calculates the total error
        error = 0
        for data_point in range(len(self.data[0])):
            self.net.pass_state(self.data[0][data_point])
            error += self.net.get_error(self.data[1][data_point])
        print(f"Error: {error}")

    # repeats the processes of gradient decent reps times
    def optimize(self, reps, step):
        error = 0
        for data_point in range(len(self.data[0])):
            self.net.pass_state(self.data[0][data_point])
            error += self.net.get_error(self.data[1][data_point])
        print("Optimization starting...")
        print(f"Step 0/{reps}")
        print(f"Error: {error}")
        for rep in range(reps):
            print(f"Step {rep+1}/{reps}")
            self.optimize_step(step)
        print("Optimization complete")
