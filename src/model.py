import numpy as np
from scipy.interpolate import interp1d

class SPGD:
    def __init__(self, Ranges, Inputs, Outputs, nFun, nModes, activeDim):
        self.nDim = Ranges.shape[1]
        self.Ranges = Ranges
        self.Inputs = Inputs
        self.nS = Inputs.shape[0]
        self.Outputs = Outputs.copy()
        if Outputs.ndim == 2:
            self.Outputs = self.Outputs
        else:
            self.Outputs = self.Outputs[:, None]
        self.nO = self.Outputs.shape[1]
        self.nFun = nFun
        self.nModes = nModes
        self.nTotModes = self.nModes
        self.N_ = 101
        self.x = np.zeros((self.N_, self.nDim))
        self.xAdim = np.linspace(-1, 1, self.N_)
        self.Bases = []
        for k in range(self.nDim):
            self.x[:, k] = np.linspace(self.Ranges[0, k], self.Ranges[1, k], self.N_)
            self.Bases.append(np.polynomial.legendre.legval(interp1d(self.x[:, k], self.xAdim, fill_value='extrapolate')(self.Inputs[:, k]), np.eye(self.nFun)).T)
        self.xBase = np.polynomial.legendre.legval(self.xAdim, np.eye(self.nFun)).T
        self.previousCoefficients = np.zeros((self.nFun, self.nDim, self.nTotModes))
        self.currentCoefficients = np.empty((self.nFun, self.nDim))
        self.previousVector = np.empty((self.nTotModes, self.nO))
        self.currentVector = np.empty(self.nO)
        self.currentGuess = np.empty((self.nS, self.nDim))
        self.rhs = np.empty(self.nS)
        self.activeDim = min(max(activeDim, 1), self.nDim)
        self.fit()

    def fit(self):
        for kMode in range(self.nModes):
            self.currentCoefficients = np.vstack((np.ones((1, self.nDim)), np.zeros((self.nFun-1, self.nDim))))
            self.currentGuess = np.ones((self.nS, self.nDim))
            vector, rhs = self.initVector()
            active = np.arange(self.nDim)
            np.random.shuffle(active)
            active = self.fixedPoint(active, vector, rhs)
            self.currentCoefficients = np.vstack((np.ones((1, self.nDim)), np.zeros((self.nFun-1, self.nDim))))
            self.currentGuess = np.ones((self.nS, self.nDim))
            self.fixedPoint(active, vector, rhs)
            self.previousVector[kMode] = self.currentVector
            self.previousCoefficients[:, :, kMode] = self.currentCoefficients
            self.Outputs -= np.tensordot(np.prod(self.currentGuess, axis=1), self.currentVector, 0)
        self.predict = lambda x: np.prod([interp1d(self.x[:, i], self.xBase, axis=0, fill_value='extrapolate')(x[:, i]) @ self.previousCoefficients[:, i] for i in range(self.nDim)], axis=0) @ self.previousVector

    def initVector(self):
        idx = np.unravel_index(np.abs(self.Outputs).argmax(), self.Outputs.shape)
        row = self.Outputs[idx[0]]
        iMax = 100
        i = 0
        tol = 1e-2
        convergence = np.inf
        while (convergence > tol and i <= iMax):
            i += 1
            magnitude = row @ row
            if magnitude > 1e-8:
                col = self.Outputs @ row / magnitude
            else:
                col = self.Outputs @ row
            magnitude = col @ col
            if magnitude > 1e-8:
                newrow = col @ self.Outputs / magnitude
            else:
                newrow = col @ self.Outputs
            convergence = np.linalg.norm(newrow-row)
            row = newrow
        self.rhs = col
        return row, col

    def fixedPoint(self, active, vector, rhs):
        iMax = 1000
        i = 0
        tol = 1e-4
        convergence = np.inf
        val = np.inf
        self.currentVector = vector
        self.rhs = rhs
        while (convergence > tol and i <= iMax):
            i += 1
            newval = self.globalIteration(active)
            convergence = np.linalg.norm(newval-val)
            val = newval
        active = np.argpartition(np.abs(np.linalg.norm(self.currentCoefficients[1:, :], axis=0)/self.currentCoefficients[0, :]), self.activeDim-1)[-self.activeDim:]
        return active

    def globalIteration(self, active):
        for dim in active:
            self.singleIteration(dim)
        guess = np.prod(self.currentGuess, axis=1)
        magnitude = guess @ guess
        if magnitude > 1e-8:
            self.currentVector = (guess @ self.Outputs) / magnitude
            self.rhs = (self.Outputs @ self.currentVector) / (self.currentVector @ self.currentVector)
        else:
            self.currentVector = guess @ self.Outputs
            self.rhs = self.Outputs @ self.currentVector
        val = np.tensordot(guess, self.currentVector, 0)
        return val

    def singleIteration(self, dim):
        maxNumFun = self.nFun
        M = np.prod(self.currentGuess[:, np.setdiff1d(np.arange(self.nDim), dim)], axis=1)[:, None] * self.Bases[dim][:, :maxNumFun]
        try:
            coef = np.linalg.solve(M.T @ M, M.T @ self.rhs)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                coef = np.zeros(self.nFun)
                coef[0] = 1
            else:
                raise
        self.currentCoefficients[:maxNumFun, dim] = coef[:maxNumFun]
        self.currentGuess[:, dim] = self.Bases[dim] @ self.currentCoefficients[:, dim]

    @property
    def N(self):
        N = []
        for d in range(self.nDim):
            N.append(interp1d(self.x[:, d], self.xBase, axis=0, fill_value='extrapolate'))
        return N

    @property
    def X(self):
        X = []
        for d in range(self.nDim):
            X.append(interp1d(self.x[:, d], self.xBase @ self.previousCoefficients[:, d], axis=0, fill_value='extrapolate'))
        return X
