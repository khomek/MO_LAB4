import sys
from PyQt5.QtWidgets import QApplication
from app_window import OptimizationUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizationUI()
    window.show()
    sys.exit(app.exec_())