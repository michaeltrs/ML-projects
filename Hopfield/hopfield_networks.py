import numpy as np
from copy import deepcopy


class Hopfield:

    def __init__(self, patterns, n=None):
        """
        patterns : list or array of flatten patterns to be stored
        n        : learning rate
        """
        self.patterns = np.array([pattern.flatten() for pattern in patterns])
        self.unique_patterns, idx, counts = np.unique(self.patterns,
            axis=0, return_index=True, return_counts=True)
        self.pattern_dim = dict(zip(idx, counts))
        self.N = np.shape(self.patterns)[1] # num dim of a pattern
        if n is None:
            self.n = 1/self.N
        else:
            self.n = n
        self.W = self.get_weights()

    def wij(self, i, j):
        return self.n * self.patterns[:, i].T.dot(self.patterns[:, j])

    def get_weights(self):
        return np.array([[self.wij(i, j) for j in range(self.N)]
                         for i in range(self.N)])

    def update(self, input_pattern, rule=0):
        """
        Update network for input
         - asynchronously, rule=0
         - synchronously , rule=1
        """
        if rule is 0:
            h = deepcopy(input_pattern)
            for i, _ in enumerate(h):
                h[i] = self.W[i, :].dot(h)
        elif rule is 1:
            h = self.W.dot(input_pattern)
        updated_pattern = np.array([self.up(hi) for hi in h])
        return updated_pattern

    @staticmethod
    def distance(s0, s1):
        """
        Hamming distance between states s0 and s1
        """
        return np.sum(np.ones(s0 == s1))

    @staticmethod
    def up(h):
        """
        Update rule
        """
        if np.sign(h) in [0, 1]:
            return 1
        else:
            return -1

    @staticmethod
    def print(x, shape):
        print(x.reshape(shape))

    def energy(self, x=None):
        if x is None:
            x = self.pattern
        return -1/2 * x.dot(self.W).dot(x)


class StochasticHopfield(Hopfield):

    def __init__(self, patterns, T, n=None):
        Hopfield.__init__(self, patterns, n)
        self.T = T

    # def flip_prob1(self, x1, x2):
    #     return 1 / (1 + np.exp((self.energy(x2) - self.energy(x1)) / self.T))
    #
    # def flip_prob2(self, x1):
    #     h = x1
    #     for i, _ in enumerate(h):
    #         h[i] = self.W[i, :].dot(h)
    #     return 1 / (1 + np.exp((-2*h*x / self.T)))

    def up(self, h):
        """
        Update rule
        
        Could implement by passing x with:
        p1 = 1 / (1 + np.exp(-2 * h * x/ self.T))
        and return x, -x
        """
        # probability of "1"
        p1 = 1 / (1 + np.exp(-2 * h / self.T))

        if np.random.rand() < p1:
            return 1
        else:
            return -1


if __name__ == "__main__":

    pattern0 = np.array([1 if i%2==0 else -1 for i in range(169)])

    all_patterns = [np.random.randn(np.shape(pattern0)[0]) for i in range(50)]

    for i in range(3):
        all_patterns.append(pattern0)#.append(pattern0)

    # network = Hopfield(all_patterns, n=None)
    # print("Energy at stored pattern: %.3f" % network.energy(pattern0))
    #
    # x0 = np.random.randn(np.shape(pattern0)[0])
    # print("Energy at random point: %.3f" % network.energy(x0))
    #
    # x = x0
    # for it in range(1, 25):
    #
    #     x = network.update(x, rule=0)
    #
    #     print("Energy at iteration %d: %.3f" % (it, network.energy(x)))
    #
    # network.print(x, (13, 13))


    network = StochasticHopfield(all_patterns, T=0.5, n=None)
    print("Energy at stored pattern: %.3f" % network.energy(pattern0))

    x0 = np.random.randn(np.shape(pattern0)[0])
    print("Energy at random point: %.3f" % network.energy(x0))

    x = x0
    for it in range(1, 25):
        x = network.update(x, rule=0)

        print("Energy at iteration %d: %.3f" % (it, network.energy(x)))

    network.print(x, (13, 13))





# network.energy(pattern2.flatten())
# network.energy(x)

# pattern0 = -np.ones(144)
# pattern1 = np.ones(144)
# pattern2 = np.array([1 if i%2==0 else -1 for i in range(144)])

# pattern0 = np.array([[0, 0, 1, 0, 0],
#                      [0, 1, 1, 1, 0],
#                      [1, 1, 1, 1, 1],
#                      [0, 1, 1, 1, 0],
#                      [0, 0, 1, 0, 0]])

# pattern0 = np.array([[-1, -1,  1, -1, -1],
#                      [-1,  1,  1,  1, -1],
#                      [ 1,  1,  1,  1,  1],
#                      [-1,  1,  1,  1, -1],
#                      [-1, -1,  1, -1, -1]]).flatten()
#
# pattern1 = np.array([[-1, -1,  1, -1, -1],
#                      [-1,  1,  1,  1, -1],
#                      [ 1,  1,  1,  1,  1],
#                      [ 1, -1, -1, -1,  1],
#                      [ 1,  1, -1,  1,  1]]).flatten()
#
# pattern2 = np.array([[ 1, -1,  1, -1,  1],
#                      [-1,  1,  1,  1, -1],
#                      [ 1,  1,  1,  1,  1],
#                      [ 1, -1, -1, -1,  1],
#                      [-1,  1, -1,  1, -1]]).flatten()
