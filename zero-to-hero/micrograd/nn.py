import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class TanhNeuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"TanhNeuron({len(self.w)})"

class QuadraticNeuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # quadratic weights
        self.v = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # linear weights
        self.b = Value(0)

    def __call__(self, x):
        quad = sum((wi*xi*xi for wi, xi in zip(self.w, x)), Value(0))
        lin = sum((vi*xi for vi, xi in zip(self.v, x)), self.b)
        return quad + lin

    def parameters(self):
        return self.w + self.v + [self.b]

    def __repr__(self):
        return f"QuadraticNeuron({len(self.w)})"

class RBFNeuron(Module):
    # f(x) = exp(-γ · ||x − c||²)
    def __init__(self, nin):
        # center of the radial basis function, one coordinate per input
        self.c = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # width parameter; > 0 keeps the exponent negative (a true "bump")
        self.gamma = Value(1.0)

    def __call__(self, x):
        assert len(x) == len(self.c), f"expected {len(self.c)} inputs, got {len(x)}"
        # squared distance ||x - c||^2, built entirely from Value ops
        dist_sq = sum(((xi - ci)**2 for xi, ci in zip(x, self.c)), Value(0))
        # f(x) = exp(-gamma * ||x - c||^2)
        return (-self.gamma * dist_sq).exp()

    def parameters(self):
        return self.c + [self.gamma]

    def __repr__(self):
        return f"RBFNeuron({len(self.c)})"

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, neuron_cls=Neuron, **kwargs):
        self.neurons = [neuron_cls(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, hidden_neuron_cls=Neuron, output_neuron_cls=Neuron):
        sz = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            is_output_layer = i == len(nouts) - 1
            neuron_cls = output_neuron_cls if is_output_layer else hidden_neuron_cls

            if neuron_cls is Neuron:
                self.layers.append(
                    Layer(sz[i], sz[i+1], neuron_cls=neuron_cls, nonlin=not is_output_layer)
                )
            else:
                self.layers.append(
                    Layer(sz[i], sz[i+1], neuron_cls=neuron_cls)
                )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"