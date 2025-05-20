import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5 import QtWidgets


class OptimizationPlot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax = self.figure.add_subplot(111, projection='3d')
        self.clear_plot()

    def update_plot(self, history, func, constraint, result_point):
        self.ax.clear()

        # Определение границ графика на основе истории и результата
        if history:
            path = np.array(history)
            x_min, x_max = path[:, 0].min() - 0.5, path[:, 0].max() + 0.5
            y_min, y_max = path[:, 1].min() - 0.5, path[:, 1].max() + 0.5
        else:
            x_min, x_max = result_point[0] - 1, result_point[0] + 1
            y_min, y_max = result_point[1] - 1, result_point[1] + 1

        # Подготовка сетки
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)

        # Вычисление функции и ограничения на сетке
        Z_func = np.zeros_like(X)
        Z_constr = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_func[i, j] = func([X[i, j], Y[i, j]])
                Z_constr[i, j] = constraint([X[i, j], Y[i, j]])


        self.ax.plot_surface(X, Y, Z_func, cmap='viridis', alpha=0.7, label='Функция')

        self.ax.plot_surface(X, Y, Z_constr, cmap='coolwarm', alpha=0.7, label='Ограничение')

        # Траектория оптимизации
        if history:
            path = np.array(history)
            z_path = [func(p) for p in path]
            self.ax.plot(path[:, 0], path[:, 1], z_path, 'r-', linewidth=2, marker='o', markersize=4,
                         label='Траектория')

        # Отображение точки минимума
        last_point_z = func(result_point)
        self.ax.scatter([result_point[0]], [result_point[1]], [last_point_z],
                        c='black', marker='o', s=100, label='Минимум')

        # Добавление аннотации
        self.ax.text(result_point[0], result_point[1], last_point_z,
                     f'Min: ({result_point[0]:.2f}, {result_point[1]:.2f}, {last_point_z:.2f})',
                     color='black', fontsize=10)

        self.ax.set_xlabel('x0')
        self.ax.set_ylabel('x1')
        self.ax.set_zlabel('Значение')
        self.ax.legend()
        self.canvas.draw()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('x0')
        self.ax.set_ylabel('x1')
        self.ax.set_zlabel('Значение')
        self.canvas.draw()