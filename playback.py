from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
import sys

class ProgressBar(QtWidgets.QWidget):
    progress_changed = pyqtSignal(int)
    
    def __init__(self, max_progress, parent=None):
        super().__init__(parent)
        self.max_progress = max_progress
        self.current_progress = 0
        self.setFixedHeight(20)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Background bar
        painter.setBrush(QtGui.QColor(80, 80, 80))
        painter.drawRect(0, 5, self.width(), 10)
        
        # Progress fill
        progress_width = int((self.current_progress / max(1, self.max_progress)) * self.width())
        painter.setBrush(QtGui.QColor(0, 200, 255))
        painter.drawRect(0, 5, progress_width, 10)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.progress_update(event.pos().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.progress_update(event.pos().x())
    
    def progress_update(self, x_pos):
        new_progress = int((x_pos / max(1, self.width())) * self.max_progress)
        old_progress = self.current_progress
        self.current_progress = max(0, min(new_progress, self.max_progress))
        if old_progress != self.current_progress:
            self.progress_changed.emit(self.current_progress)
        self.update()

    def update_progress(self, progress):
        self.current_progress = progress
        self.update()

class PlaybackUI(QtWidgets.QWidget):
    def __init__(self, max_progress, pause_resume_callback, bar_callback):
        super().__init__()
        self.max_progress = max_progress
        
        self.setWindowTitle("Playback Control")
        self.setGeometry(300, 100, 400, 0)  # Set specific width, automatic height
        self.setStyleSheet("background-color: black;")
        
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        
        self.progress_bar = ProgressBar(self.max_progress)
        self.progress_bar.progress_changed.connect(bar_callback)
        
        self.progress_label = QLabel(f"0/{self.max_progress}")
        self.progress_label.setStyleSheet("color: white;")
        self.progress_label.setFixedWidth(80)
        
        # Horizontal layout for label and progress bar
        self.progress_layout = QHBoxLayout()
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.setStretch(0, 0)  # Label does not stretch
        self.progress_layout.setStretch(1, 1)  # Bar takes remaining space but not beyond
        
        self.layout.addLayout(self.progress_layout)
        
        # Pause/Resume button
        self.pause_resume_button = QPushButton("Pause/Resume")
        self.pause_resume_button.setStyleSheet("background-color: white; color: black;")
        self.pause_resume_button.clicked.connect(pause_resume_callback)
        self.layout.addWidget(self.pause_resume_button)
        
        self.setLayout(self.layout)
        self.adjustSize()  # Automatically adjust height based on content
    
    def update_progress(self, current):
        self.progress_bar.update_progress(current)
        self.progress_label.setText(f"{current}/{self.max_progress}")

    def update_state(self, paused):
        self.pause_resume_button.setText("Resume" if paused else "Pause")
    
    def width(self, w):
        self.setMinimumWidth(w)

if __name__ == "__main__":
    def test():
        window.state_update(False)
        window.progress_update(50)
    def test2(new_pos):
        print("asdf", new_pos)
    app = QtWidgets.QApplication(sys.argv)
    window = PlaybackUI(6039, test, test2)
    window.update_progress(1000)
    window.show()
    sys.exit(app.exec_())
