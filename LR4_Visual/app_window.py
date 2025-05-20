from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QComboBox, QTextEdit)
from PyQt5.QtCore import Qt
from plot_window import OptimizationPlot
from BarierMethod import BarrierMethod
from PenaltyMethod import PenaltyMethod
import numpy as np


class OptimizationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Методы оптимизации с ограничениями")
        self.resize(1200, 800)

        # Инициализация методов
        self.barrier_method = BarrierMethod()
        self.penalty_method = PenaltyMethod()

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель управления
        control_panel = QGroupBox("Параметры оптимизации")
        control_layout = QVBoxLayout()

        # Выбор метода
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Метод барьерных функций", "Метод штрафов"])

        # Параметры
        self.x0_input = self.create_input_field(control_layout, "Первое значение начального x0:", "0.5")
        self.x1_input = self.create_input_field(control_layout, " Второе значение начального x0:", "0.5")
        self.r0_input = self.create_input_field(control_layout, "Начальный r0:", "1.0")
        self.C_input = self.create_input_field(control_layout, "Коэффициент C:", "5.0")
        self.eps_input = self.create_input_field(control_layout, "Точность eps:", "0.05")

        # Кнопки
        self.run_btn = QPushButton("Запустить оптимизацию")
        self.clear_btn = QPushButton("Очистить")

        # Вывод результатов
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        # Сборка панели управления
        control_layout.addWidget(QLabel("Метод оптимизации:"))
        control_layout.addWidget(self.method_combo)
        control_layout.addWidget(self.run_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(QLabel("Результаты:"))
        control_layout.addWidget(self.result_display)
        control_panel.setLayout(control_layout)

        # Правая панель с графиком
        self.plot_widget = OptimizationPlot()

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.plot_widget)

    def create_input_field(self, layout, label_text, default_value=""):
        layout.addWidget(QLabel(label_text))
        line_edit = QLineEdit(default_value)
        layout.addWidget(line_edit)
        return line_edit

    def connect_signals(self):
        self.run_btn.clicked.connect(self.run_optimization)
        self.clear_btn.clicked.connect(self.clear_results)

    def run_optimization(self):
        try:
            # Получение параметров
            x0 = [float(self.x0_input.text()), float(self.x1_input.text())]
            r0 = float(self.r0_input.text())
            C = float(self.C_input.text())
            eps = float(self.eps_input.text())

            # Выбор метода
            if self.method_combo.currentText() == "Метод барьерных функций":
                optimizer = self.barrier_method
                result, iterations = optimizer.barrier_method(x0=x0, r0=r0, C=C, eps=eps)
                history = optimizer.history
            else:
                optimizer = self.penalty_method
                result, iterations = optimizer.penalty_method(x0=x0, r0=r0, C=C, eps=eps)
                history = optimizer.history

            # Вывод результатов
            result_text = (
                f"Метод: {self.method_combo.currentText()}\n"
                f"Результат: x = [{result[0]:.5f}, {result[1]:.5f}]\n"
                f"Значение функции: {optimizer.func(result):.5f}\n"
                f"Количество итераций: {iterations}"
            )
            self.result_display.setPlainText(result_text)

            # Обновление графика
            self.plot_widget.update_plot(
                history=history,
                func=optimizer.func,
                constraint=optimizer.constraint if hasattr(optimizer, 'constraint') else optimizer.limitation,
                result_point=result
            )

        except Exception as e:
            self.result_display.setPlainText(f"Ошибка: {str(e)}")

    def clear_results(self):
        self.result_display.clear()
        self.plot_widget.clear_plot()