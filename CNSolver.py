"""MIT License

Copyright (c) [2024] [Oluwafolajimi Dere]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from scipy.linalg import lu_solve, lu_factor
from scipy.interpolate import interp1d


class CNSolver:
    """A Crank-Nicolson Finite Difference solver for pricing European options.

    Prices European options at all times to maturity by numerical solution of
    the Black-Scholes equation on a 2-D domain and calculates Greeks surfaces
    by numerical differentiation. Option value and greeks for the current
    underlying price are compiled in results.

    Typical usage example:

    eur_opts = CNSolver(S0, K, T, sigma, r)
    eur_opts = CNSolver.solve()
    CNSolver.results()
    """

    def __init__(self, S0, K, T, sigma, r):
        """Initialises class instance with Black-Scholes model parameters.

        Initialises class instance with Black-Scholes model parameters and sets
        grid and domain for numerical scheme. Numerical method is set to manual
        implementation of PLU decomposition and forwards/backwards substitution
        solver by default.

        Args:
            S0 (float): Current underlying market price.
            K (float): Strike price of option.
            T (float): Time to maturity in years of contract.
            sigma (float): Annualised volatility of underlying market.
            r (float): Risk-free interest rate in underlying market.
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.N = 1200
        self.M = 600
        self.Smax = 3 * S0
        self.dS = self.Smax / self.M
        self.dt = self.T / self.N
        self.manual = True
        self.S = np.linspace(0, self.Smax, 1 + self.M)
        self.Ts = np.linspace(0, self.T, 1 + self.N)

        # Attributes defined in later initialisation methods

        self.P = None
        self.C = None
        self.P_price = None
        self.P_delta = None
        self.P_gamma = None
        self.P_theta = None
        self.C_price = None
        self.C_delta = None
        self.C_gamma = None
        self.C_theta = None
        self.P_delta_func = None
        self.P_gamma_func = None
        self.P_theta_func = None
        self.C_delta_func = None
        self.C_gamma_func = None
        self.C_theta_func = None
        self.A = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.piv = None
        self.LU = None
        self.L = None
        self.U = None
        self.D = None
        self.d = None

    def solve(self):
        """Solves Black-Scholes equation.

        Solves Black-Scholes equation to price both European Call and Put
        options. Interpolation of present values is done for displaying
        results.
        """
        self.init_matrix()
        self.set_coeffs()
        self.solveLinearSystem()
        self.calculate_call()
        self.calculate_greeks()

        # interpolation
        self.P_price = interp1d(
            self.S.flatten(),
            self.P[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.C_price = interp1d(
            self.S.flatten(),
            self.C[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.P_delta_func = interp1d(
            self.S.flatten(),
            self.P_delta[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.C_delta_func = interp1d(
            self.S.flatten(),
            self.C_delta[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.P_gamma_func = interp1d(
            self.S.flatten(),
            self.P_gamma[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.C_gamma_func = interp1d(
            self.S.flatten(),
            self.C_gamma[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.P_theta_func = interp1d(
            self.S.flatten(),
            self.P_theta[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )
        self.C_theta_func = interp1d(
            self.S.flatten(),
            self.C_theta[:, 0].flatten(),
            kind="linear",
            fill_value="extrapolate",
        )

    def init_matrix(self):
        """Initialises solution matrix for option value.

        Initialises M+1 x N+1 matrix with appropriate European Put boundary
        conditions to store the option value during the numerical scheme. Grid
        spacing is also calculated.
        """
        N = self.N
        M = self.M
        self.P = np.zeros((M + 1, N + 1))
        K = self.K
        r = self.r
        dS = self.dS
        dt = self.dt

        # European Put boundaries

        for n in range(N):
            self.P[0, n] = K * np.exp(-r * (N - n) * dt)

        for m in range(M):
            self.P[m, N] = np.max([K - m * dS, 0])

        self.P[M, :] = 0

    def set_coeffs(self):
        """Initialises coefficient matrices for Crank-Nicolson finite difference scheme.

        Initialises coefficient matrices for Crank-Nicolson finite difference scheme.
        PLU decomposition is used to transform the system Ax[i] = Dx[i-1] + b into
        PLUx[i] = Dx[i-1] + b to speed up numerical solution. LU decomposition is done
        using SciPy package if the manual option is switched off.
        """
        M = self.M
        dt = self.dt
        sigma = self.sigma
        r = self.r
        ms = np.arange(0, M + 1)
        self.alpha = 0.25 * dt * ((sigma * ms) ** 2 - r * ms)
        self.beta = -0.5 * dt * ((sigma * ms) ** 2 + r)
        self.gamma = 0.25 * dt * ((sigma * ms) ** 2 + r * ms)
        self.A = (
            -np.diag(self.alpha[2:M], -1)
            + np.diag(1 - self.beta[1:M])
            - np.diag(self.gamma[1 : M - 1], 1)
        )

        if self.manual:
            self.piv, self.L, self.U = self.lu_factor(self.A)
        else:
            self.LU, self.piv = lu_factor(self.A)

        self.D = (
            np.diag(self.alpha[2:M], -1)
            + np.diag(1 + self.beta[1:M])
            + np.diag(self.gamma[1 : M - 1], 1)
        )
        self.d = np.zeros_like(self.D[0])

    def solveLinearSystem(self):
        """Solves PLUx = b"""
        N = self.N
        M = self.M
        d = self.d
        alpha = self.alpha
        gamma = self.gamma
        P = self.P
        D = self.D
        for n in range(N - 1, -1, -1):
            d[0] = alpha[1] * (P[0, n] + P[0, n + 1])
            d[-1] = gamma[-1] * (P[-1, n] + P[-1, n + 1])

            if self.manual:
                piv = self.piv
                L = self.L
                U = self.U
                P[1:M, n] = self.lu_solve(piv, L, U, D @ P[1:M, n + 1] + d)
            else:
                LU = self.LU
                piv = self.piv
                P[1:M, n] = lu_solve((LU, piv), D @ P[1:M, n + 1] + d)

    def lu_factor(self, A):
        """Performs PLU decomposition of matrix A

        Factorises matrix A into P, L and U of PLU decomposition.

        Args:
            A (numpy.array): Input matrix.

        Returns:
            P (numpy.array): Pivot matrix in PLU decomposition.
            L (numpy.array): Lower triangular matrix in PLU decomposition.
            U (numpy.array): Upper triangular matrix in PLU decomposition.
        """
        n = A.shape[0]
        L = np.eye(n)
        P = np.eye(n)
        U = A.copy()

        for i in range(n):
            k = i

            # Swaps rows of P and U if diagonal entries are zero
            for k in range(i, n):
                if ~np.isclose(U[i, i], 0.0):
                    break
                U[k, k + 1] = U[k + 1, k]
                P[k, k + 1] = P[k + 1, k]

            # Remove entries below i with row operations on U
            # Reverse the operations on L
            for j in range(i + 1, n):
                l = U[j, i] / U[i, i]
                L[j, i] = l
                U[j, :] -= l * U[i, :]

        return P, L, U

    def lu_solve(self, P, L, U, b):
        """Algorithm to solve linear system by concurrent forwards/backwards substitution.

        Solves the linear system PLUv[i] = Dv[i-1] + b by solving Ly = P @ (Dv[i-1] + b)
        by forwards substitution, and then solving Ux = y by backwards substitution. This
        is the default method for solving the Black-Scholes equation using the Crank-Nicolson
        finite difference scheme. It is the manual implementation of the scipy.linalg.lu_solve.

        Args:
            P (numpy.array): Pivot matrix in PLU decomposition.
            L (numpy.array): Lower triangular matrix in PLU decomposition.
            U (numpy.array): Upper triangular matrix in PLU decomposition.
            b (numpy.array): Vector in linear system.

        Returns:
            x (numpy.array): Solution vector for linear system.
        """

        # solve Ly = Pb

        n = L.shape[0]

        y = np.zeros_like(b)
        pb = np.dot(P, b)

        y[0] = pb[0] / L[0, 0]

        for i in range(1, n):
            y[i] = (pb[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

        # solve Ux = y

        x = np.zeros_like(y)

        x[-1] = y[-1] / U[-1, -1]

        for i in range(n - 2, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

        return x

    def calculate_call(self):
        """Calculates European Call option value surface from European Put surface.

        Uses the put-call parity between European Call and Put options to derive the
        value surface for the European Call option with the same parameters as the
        Put.
        """
        N = self.N
        M = self.M
        K = self.K
        r = self.r
        dS = self.dS
        dt = self.dt
        P = self.P
        self.C = np.zeros_like(P)

        for n in range(N + 1):
            for m in range(M + 1):
                self.C[m, n] += m * dS + P[m, n] - K * np.exp(-r * (N - n) * dt)

    def calculate_greeks(self):
        """Calculates European Call and Put option Greeks surfaces.

        Calculates Delta, Gamma and Theta surfaces for European Call and Put
        """
        self.calculate_delta()
        self.calculate_gamma()
        self.calculate_theta()

    def calculate_delta(self):
        """Calculates European Call and Put Delta surfaces.

        Calculates European Call and Put Delta surfaces by first-order differencing.
        """
        N = self.N
        M = self.M
        P = self.P
        C = self.C
        dS = self.dS
        P_delta = P.copy()
        C_delta = C.copy()

        for n in range(N + 1):
            # first order backwards and forwards differences
            P_delta[0, n] = (P[1, n] - P[0, n]) / dS
            P_delta[M - 1, n] = (P[M - 1, n] - P[M - 2, n]) / dS
            C_delta[0, n] = (C[1, n] - C[0, n]) / dS
            C_delta[M - 1, n] = (C[M - 1, n] - C[M - 2, n]) / dS
            for m in range(1, M):
                P_delta[m, n] = (P[m + 1, n] - P[m - 1, n]) / (2 * dS)
                C_delta[m, n] = (C[m + 1, n] - C[m - 1, n]) / (2 * dS)

        # Workaround for discontinuity in European Call option values

        C_delta[M, :] = C_delta[M - 1, :]

        self.P_delta = P_delta
        self.C_delta = C_delta

    def calculate_gamma(self):
        """Calculates European Call and Put Gamma surfaces.

        Calculates European Call and Put Gamma surfaces by second-order differencing.
        Divergent values towards maturity are smoothed out for display purposes.
        """
        N = self.N
        M = self.M
        P = self.P
        C = self.C
        dS = self.dS
        P_gamma = P.copy()
        C_gamma = C.copy()

        for n in range(N + 1):
            P_gamma[0, n] = (P[2, n] - 2 * P[1, n] + P[0, n]) / (dS**2)
            P_gamma[M - 1, n] = (P[M - 1, n] - 2 * P[M - 2, n] + P[M - 3, n]) / (dS**2)
            C_gamma[0, n] = (C[2, n] - 2 * C[1, n] + C[0, n]) / (dS**2)
            C_gamma[M - 1, n] = (C[M - 1, n] - 2 * C[M - 2, n] + C[M - 3, n]) / (dS**2)
            for m in range(1, M):
                P_gamma[m, n] = (P[m + 1, n] - 2 * P[m, n] + P[m - 1, n]) / (dS**2)
                C_gamma[m, n] = (C[m + 1, n] - 2 * C[m, n] + C[m - 1, n]) / (dS**2)

        # Smoothing of near maturity values

        self.smooth_surface(P_gamma)
        self.smooth_surface(C_gamma)

        # Workaround for prevailing discontinuity in European Call option values

        C_gamma[M, :] = C_gamma[M - 1, :]

        self.P_gamma = P_gamma
        self.C_gamma = C_gamma

    def calculate_theta(self):
        """Calculates European Call and Put Theta surfaces.

        Calculates European Call and Put Theta surfaces by first-order differencing.
        Divergent values towards maturity are smoothed out for display purposes.
        """
        N = self.N
        M = self.M
        P = self.P
        C = self.C
        dt = self.dt
        P_theta = P.copy()
        C_theta = C.copy()

        for m in range(M + 1):
            P_theta[m, 0] = (P[m, 1] - P[m, 0]) / dt
            P_theta[m, N - 1] = (P[m, N - 1] - P[m, N - 2]) / dt
            C_theta[m, 0] = (C[m, 1] - C[m, 0]) / dt
            C_theta[m, N - 1] = (C[m, N - 1] - C[m, N - 2]) / dt
            for n in range(1, N - 1):
                P_theta[m, n] = (P[m, n + 1] - P[m, n - 1]) / (2 * dt)
                C_theta[m, n] = (C[m, n + 1] - C[m, n - 1]) / (2 * dt)

        # Smoothing of near maturity values

        self.smooth_surface(P_theta)
        self.smooth_surface(C_theta)

        self.P_theta = P_theta
        self.C_theta = C_theta

    def smooth_surface(self, A):
        """Smoothing surface at times close to maturity.

        Aggressive smoothing scheme for surface values close to maturity.
        Implemented to deal with discontinuities in greeks at t=T. Operation
        done in-place to reduce memory demand.

        Args:
            A (numpy.array): Input matrix.

        Returns:
            A (numpy.array): Smoothed output matrix.
        """
        for n in range(-30, 0, 1):
            A[:, n] = (
                0.1 * A[:, n]
                + 0.2 * A[:, n - 1]
                + 0.3 * A[:, n - 2]
                + 0.4 * A[:, n - 3]
            )
        for n in range(-30, 0, 1):
            A[:, n] = 0.1 * A[:, n - 1] + 0.2 * A[:, n - 2] + 0.7 * A[:, n - 3]

        return A

    def results(self):
        """Prints European Call and Put option present value and greeks at current underlying market price.

        Prints European Call and Put option present value and greeks
        at current underlying market price. Indicates whether manual
        computation was used to calculate values.
        """
        print("European Call Option\n")
        print("Value: " + str(self.C_price(self.S0)))
        print("Delta: " + str(self.C_delta_func(self.S0)))
        print("Gamma: " + str(self.C_gamma_func(self.S0)))
        print("Theta: " + str(self.C_theta_func(self.S0)))
        print("\nEuropean Put Option\n")
        print("Value: " + str(self.P_price(self.S0)))
        print("Delta: " + str(self.P_delta_func(self.S0)))
        print("Gamma: " + str(self.P_gamma_func(self.S0)))
        print("Theta: " + str(self.P_theta_func(self.S0)))

        if self.manual:
            print("\nManual LU Decomposition and Linear Solver used.")
        else:
            print("\nSciPy linalg lu_factor and lu_solve used.")


if __name__ == "__main__":
    # unit test one for CNSolver.py

    # set model parameters
    S0 = 50
    K = 60
    T = 1.5
    sigma = 0.2
    r = 0.1

    # initialise solver and solve using manual implementation then using SciPy
    solver = CNSolver(S0, K, T, sigma, r)
    solver.solve()
    solver.results()
    solver.manual = False
    print("---------------------------")
    solver.solve()
    solver.results()
