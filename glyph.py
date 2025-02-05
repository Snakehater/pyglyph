from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
import sys

class GlyphLED(QtWidgets.QLabel):
    def __init__(self, width, height, position, parent, rotation=0):
        super().__init__(parent)
        self.width_value = width  # Store width
        self.height_value = height  # Store height
        self.rotation = rotation  # Store rotation angle
        self.setFixedSize(*self.get_adjusted_size())  # Adjust size to fit rotated shape
        self.brightness = 0x70
        self.move(*self.get_adjusted_position(position))
        self.update()
        #self.mousePressEvent = self.toggle_led  # Add click event

    def get_adjusted_size(self):
        if self.rotation % 180 != 0:
            diagonal = (self.width_value ** 2 + self.height_value ** 2) ** 0.5
            return int(diagonal), int(diagonal)  # Ensure enough space for rotation
        return self.width_value, self.height_value

    def get_adjusted_position(self, position):
        adjusted_size = self.get_adjusted_size()
        return position[0] - (adjusted_size[0] - self.width_value) // 2, position[1] - (adjusted_size[1] - self.height_value) // 2
    
    def set_led(self, value):
        self.brightness = value
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.rotation)
        painter.translate(-self.width() / 2, -self.height() / 2)
        
        color = QtGui.QColor(f"#{self.brightness:x}ffffff")
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        
        rect = QtCore.QRect((self.width() - self.width_value) // 2, (self.height() - self.height_value) // 2, self.width_value, self.height_value)
        painter.drawRoundedRect(rect, min(self.width_value, self.height_value) // 2, min(self.width_value, self.height_value) // 2)

class GlyphUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nothing Phone Glyph LEDs")
        self.setGeometry(300, 100, 300, 600)
        self.setStyleSheet("background-color: black;")

        # Define LED positions and sizes based on the requested shape
        self.leds = []
        led_positions = [
            (50, 50, (30, 30)),  # Top camera Circle
            (80, 10, (180, 50), -45),  # Top Right Diagonal Line (rotated 45 degrees)
            (80, 10, (30, 290), 45),  # circle bottom left
            (80, 10, (180, 290), -45),  # circle bottom right
            (80, 10, (180, 150), 45),  # cricle top right
            (80, 10, (30, 150), -45),  # circle top left
            (15, 15, (137, 450)),  # Dot
            (10, 10, (140, 420)),  # Vertical Line
            (10, 10, (140, 410)),  # Vertical Line
            (10, 10, (140, 400)),  # Vertical Line
            (10, 10, (140, 390)),  # Vertical Line
            (10, 10, (140, 380)),  # Vertical Line
            (10, 10, (140, 370)),  # Vertical Line
            (10, 10, (140, 360)),  # Vertical Line
            (10, 10, (140, 350)),  # Vertical Line
        ]

        for led in led_positions:
            if len(led) == 4:
                width, height, position, rotation = led
                led = GlyphLED(width, height, position, self, rotation)
            else:
                width, height, position = led
                led = GlyphLED(width, height, position, self)
            self.leds.append(led)
    def set_leds(self, idx, value):
        self.leds[idx].set_led(value if value > 0x10 else 0x10)
        self.leds[idx].update()
    def glyph_update(self, data):
        if len(data) != 15:
            return
        for i in range(15):
            self.set_leds(i, data[i])

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = GlyphUI()
    window.show()
    sys.exit(app.exec_())
