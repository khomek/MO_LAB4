import numpy as np

class BarrierMethod:
    def __init__(self):
        self.history = []

    def func(self, x):
        return 2 * x[0]**2 + x[1]**2

    def constraint(self, x):
        return 2 - 2*x[0] - x[1]

    def barrier_function(self, x, r):
        if self.constraint(x) <= 0:
            return np.inf
        return self.func(x) - r * np.log(self.constraint(x)) #логарифмическая функция

    def deriv_barrier(self, x, r):

        df_dx1 = 4 * x[0] + (2*r) / (2*x[0] + x[1] - 2)
        df_dx2 = 2 * x[1] + (r) / (2*x[0] + x[1] - 2)
        return np.array([df_dx1, df_dx2])

    def penalty_value(self, x, r):
      return - r * np.log(self.constraint(x))

    def norma(self, vector):
        return np.sqrt(vector[0]**2 + vector[1]**2)

    def constant_step_gradient(self, x0, r, t=0.1, eps1=0.15, eps2=0.2, M=10):

        x = np.array(x0, dtype=float)
        x_prev = x.copy()
        k = 0
        fl = False
        barrier_func = lambda x_val: self.barrier_function(x_val, r)

        while True:
            if self.constraint(x) <= 0:
                return x_prev
            gradient = self.deriv_barrier(x, r)
            grad_norm = self.norma(gradient)

            if grad_norm < eps1 or k >= M:
                return x

            new_x = x - t * gradient
            fnew_x = barrier_func(new_x)
            fx = barrier_func(x)


            while fnew_x > fx:
                t /= 2
                new_x = x - t * gradient
                fnew_x = barrier_func(new_x)
                fx = barrier_func(x)

            if self.norma(new_x - x) < eps2 and abs(fnew_x - fx) < eps2:
                if fl:
                    x = new_x
                    break
                else:
                    fl = True

            else:
                fl = False
            x_prev = x.copy()
            x = new_x
            k += 1
        return x

    def barrier_method(self, x0=[0.5, 0.5], r0=1, C=5, eps=0.05):

        r = r0
        x = np.array(x0)
        k = 0

        while True:
            x_star = self.constant_step_gradient(x, r)

            penalty_value = abs(self.penalty_value(x_star, r))  #P(x*(r^k),r^k)



            if penalty_value < eps:
                print(f" x = {x_star}")
                print(f"f(x) = {self.func(x_star)}")
                print(f"Number of iterations: {k + 1}")
                return x_star,k+1

            else:
                r = r / C
                x = x_star
                k += 1


if __name__ == "__main__":
    obj = BarrierMethod()
    result = obj.barrier_method()
