import numpy as np

class PenaltyMethod:
    def __init__(self):
        self.history = []

    def func(self, x):
        return 2 * x[0]**2 + x[1]**2

    def limitation(self, x):
        return (2*x[0]+x[1]-2)

    def support_function(self, x, r):
        return self.func(x) + (r/2) * (self.limitation(x))**2

    def deriv_sup(self, x, r):
        df_dx1 = 4*x[0] + 2*r*(self.limitation(x))
        df_dx2 = 2*x[1] + r*(self.limitation(x))
        return np.array([df_dx1, df_dx2])

    def fine_function(self, x, r):
        return (r/2) * (self.limitation(x))**2

    def grad(self, x):
        return np.array([4 * x[0], 2 * x[1]])

    def norma(self, vector):
        return np.sqrt(vector[0]**2 + vector[1]**2)

    def constant_step_gradient_descent(self, x0, r, t=0.5, eps1=0.15, eps2=0.2, M=10):

        x = np.array(x0, dtype=float)
        k = 0
        fl = False
        support_func = lambda x_val: self.support_function(x_val, r)
        while True:
            gradient = self.deriv_sup(x,r)
            grad_norm = self.norma(gradient)

            if grad_norm < eps1 or k >= M:
                return x

            new_x = x - t * gradient
            fnew_x = support_func(new_x)
            fx = support_func(x)


            while fnew_x - fx >= 0:
                t /= 2
                new_x = x - t * gradient
                fnew_x = support_func(new_x)
                fx = support_func(x)

            if self.norma(new_x - x) < eps2 and abs(fnew_x - fx) < eps2:
                if fl:
                    x = new_x
                    break
                else:
                    fl = True

            else:
                fl = False
            x = new_x
            k += 1
        return x

    def penalty_method(self, x0=[1.5, 0.5], r0=1, C=5, eps=0.05):
        r = r0
        x = np.array(x0)
        k = 0
        while True:
            x_star = self.constant_step_gradient_descent(x, r)

            penalty_value = self.fine_function(x_star, r)


            if penalty_value < eps:
                print(f" x = {x_star}")
                print(f"f(x) = {self.func(x_star)}")
                print(f"Number of iterations: {k + 1}")
                return x_star, k+1

            else:
                r = C * r
                x = x_star
                k += 1


if __name__ == "__main__":
    obj = PenaltyMethod()
    result = obj.penalty_method()