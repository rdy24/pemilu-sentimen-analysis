import numpy as np
import random
import pickle


class Svm:
    def __init__(self):
        self.alpha = None
        self.b = None
        self.D = None
        self.data = None
        self.kernel = None
        self.kernelResults = None
        self.kernelType = None
        self.labels = None
        self.N = None
        self.usew_ = None
        self.w = None

    def train(self, data, labels, options=None):
        if options is None:
            options = {}

        # we need these in helper functions
        self.data = data
        self.labels = labels

        # parameters
        # C value. Decrease for more regularization
        C = options.get('C', 1.0)
        # numerical tolerance. Don't touch unless you're pro
        tol = options.get('tol', 1e-4)
        # non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
        alphatol = options.get('alphatol', 1e-7)
        # max number of iterations
        maxiter = options.get('maxiter', 10000)
        # how many passes over data with no change before we halt? Increase for more precision.
        numpasses = options.get('numpasses', 20)

        # instantiate kernel according to options. kernel can be given as string or as a custom function
        self.kernel = self.linear_kernel
        self.kernelType = 'linear'

        if 'kernel' in options:
            if isinstance(options['kernel'], str):
                # kernel was specified as a string. Handle these special cases appropriately
                if options['kernel'] == 'linear':
                    self.kernel = self.linear_kernel
                    self.kernelType = 'linear'

            if callable(options['kernel']):
                # assume kernel was specified as a function. Let's just use it
                self.kernel = options['kernel']
                self.kernelType = 'custom'

        # initializations
        self.N = len(data)
        self.D = len(data[0])
        self.alpha = np.zeros(self.N)
        self.b = 0.0
        self.usew_ = False  # internal efficiency flag

        # Cache kernel computations to avoid expensive recomputation.
        # This could use too much memory if N is large.
        if options.get('memoize', False):
            self.kernelResults = np.zeros((self.N, self.N))

            for i in range(self.N):
                for j in range(self.N):
                    self.kernelResults[i, j] = self.kernel(data[i], data[j])

        # run SMO algorithm
        iter = 0
        passes = 0

        while passes < numpasses and iter < maxiter:
            alphaChanged = 0

            for i in range(self.N):
                Ei = self.margin_one(data[i]) - labels[i]

                if (labels[i] * Ei < -tol and self.alpha[i] < C) or (labels[i] * Ei > tol and self.alpha[i] > 0):
                    # alpha_i needs updating! Pick a j to update it with
                    j = i

                    while j == i:
                        j = random.randint(0, self.N - 1)

                    Ej = self.margin_one(data[j]) - labels[j]

                    # calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
                    ai = self.alpha[i]
                    aj = self.alpha[j]
                    L = 0
                    H = C

                    if labels[i] == labels[j]:
                        L = max(0, ai + aj - C)
                        H = min(C, ai + aj)
                    else:
                        L = max(0, aj - ai)
                        H = min(C, C + aj - ai)

                    if abs(L - H) < 1e-4:
                        continue

                    eta = 2 * self.kernel_result(i, j) - self.kernel_result(i, i) - self.kernel_result(j, j)

                    if eta >= 0:
                        continue

                    # compute new alpha_j and clip it inside [0 C]x[0 C] box
                    # then compute alpha_i based on it.
                    newaj = aj - (labels[j] * (Ei - Ej) / eta)

                    if newaj > H:
                        newaj = H

                    if newaj < L:
                        newaj = L

                    if abs(aj - newaj) < 1e-4:
                        continue

                    self.alpha[j] = newaj
                    newai = ai + labels[i] * labels[j] * (aj - newaj)
                    self.alpha[i] = newai

                    # update the bias term
                    b1 = self.b - Ei - labels[i] * (newai - ai) * self.kernel_result(i, i) - labels[j] * (
                                newaj - aj) * self.kernel_result(i, j)

                    b2 = self.b - Ej - labels[i] * (newai - ai) * self.kernel_result(i, j) - labels[j] * (
                                newaj - aj) * self.kernel_result(j, j)

                    self.b = 0.5 * (b1 + b2)

                    if 0 < newai < C:
                        self.b = b1

                    if 0 < newaj < C:
                        self.b = b2

                    alphaChanged += 1

            iter += 1

            passes = passes + 1 if alphaChanged == 0 else 0

        # if the user was using a linear kernel, lets also compute and store the
        # weights. This will speed up evaluations during testing time
        if self.kernelType == 'linear':
            # compute weights and store them
            self.w = np.zeros(self.D)

            for j in range(self.D):
                s = 0.0

                for i in range(self.N):
                    s += self.alpha[i] * labels[i] * data[i][j]

                self.w[j] = s
                self.usew_ = True
        else:
            # okay, we need to retain all the support vectors in the training data,
            # we can't just get away with computing the weights and throwing it out

            # But! We only need to store the support vectors for evaluation of testing
            # instances. So filter here based on self.alpha[i]. The training data
            # for which self.alpha[i] = 0 is irrelevant for future.
            newdata = []
            newlabels = []
            newalpha = []

            for i in range(self.N):
                if self.alpha[i] > alphatol:
                    newdata.append(data[i])
                    newlabels.append(labels[i])
                    newalpha.append(self.alpha[i])

            # store data and labels
            self.data = newdata
            self.labels = newlabels
            self.alpha = newalpha
            self.N = len(self.data)

        trainstats = {'iters': iter}

        return trainstats

    # inst is an array of length D. Returns margin of given example
    # this is the core prediction function. All others are for convenience mostly
    # and end up calling this one somehow.
    def margin_one(self, inst):
        f = self.b

        # if the linear kernel was used and w was computed and stored,
        # (i.e. the svm has fully finished training)
        # the internal class variable usew_ will be set to true.
        if self.usew_:
            # we can speed this up a lot by using the computed weights
            # we computed these during train(). This is significantly faster
            # than the version below
            for j in range(self.D):
                f += inst[j] * self.w[j]
        else:
            for i in range(self.N):
                kernel = self.kernel
                f += self.alpha[i] * self.labels[i] * kernel(inst, self.data[i])

        return f

    def predict_one(self, inst):
        return 1 if self.margin_one(inst) > 0 else -1

    # data is an NxD array. Returns array of margins.
    def margins(self, data):
        # go over support vectors and accumulate the prediction.
        N = len(data)
        margins = np.zeros(N)

        for i in range(N):
            margins[i] = self.margin_one(data[i])

        return margins

    def kernel_result(self, i, j):
        if self.kernelResults is not None:
            return self.kernelResults[i, j]

        kernel = self.kernel

        return kernel(self.data[i], self.data[j])

    # data is NxD array. Returns array of 1 or -1, predictions
    def predict(self, data):
        margs = self.margins(data)

        for i in range(len(margs)):
            margs[i] = 1 if margs[i] > 0 else -1

        return margs

    def linear_kernel(self, v1, v2):
        s = 0

        for q in range(len(v1)):
            s += v1[q] * v2[q]

        return s

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
