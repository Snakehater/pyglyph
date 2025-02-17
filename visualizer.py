import pyaudio
import wave
import numpy as np
import argparse
import threading
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import threading
from glyph import GlyphUI
from fft import precompute_fft
from fft import glyph_compute

def audio_visualizer(wav_file):
    print("Running optimized audio visualizer...")
    
    # Precompute FFT for visualization
    fft_results, timestamps, frame_rate, chunk_size = precompute_fft(wav_file, chunk_size=2048*1, overlap=0.5)

    region_selector = [0]
    bands_lock = [False]
    fontsize = 20

    bands = [
        (462,2753,1,"lin"),
        (462,2753,1,"lin"),
        (20,277,4,"tht", 1024*3),
        (20,277,9, "bar")
    ]
    glyph_data = [glyph_compute(bands, fft_results, timestamps, frame_rate, chunk_size, simulation=True)]

    # Open audio stream
    wf = wave.open(wav_file, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Initialize PyQtGraph UI
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Real-Time Frequency Spectrum")
    win.resize(1200, 800)
    plot = win.addPlot(title="Frequency Spectrum")
    plot.setLogMode(x=True, y=False)  # Log scale for frequency
    plot.setLabel('bottom', "Frequency (Hz)")
    plot.setLabel('left', "Magnitude")
    plot.setRange(yRange=(0, 20000000), xRange=(0, 22000))  # Adjust range based on your data

    glyph_win = GlyphUI()
    glyph_win_selector = GlyphUI()

    glyph_win.move(300, 100)
    win.move(650, 100)
    glyph_win_selector.move(1850, 100)

    # Add keyboard shortcut for "Q" to quit the application
    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), win)
    def kill_session():
        stream.stop_stream()
        stream.close()
        p.terminate()
        exit()
    shortcut.activated.connect(kill_session)

    # Prepare frequency axis
    x = np.fft.fftfreq(chunk_size, d=1/frame_rate)[:chunk_size//2]
    curve = plot.plot(x, np.zeros_like(x), pen="r")
    
    # Set X range and LOCK it
    plot.setXRange(0, 5, padding=0)
    plot.vb.setLimits(xMin=0, xMax=5)

    def sync_bands():
        glyph_data[0] = glyph_compute(bands, fft_results, timestamps, frame_rate, chunk_size, simulation=True)

    def update_glyph_selector_win():
        offset = sum(band[2] for band in bands[:region_selector[0]])
        glyph_data_selected = [0]*15
        glyph_data_selected[offset:offset+bands[region_selector[0]][2]] = [255]*bands[region_selector[0]][2]
        glyph_win_selector.glyph_update(glyph_data_selected)

    def update_band():
        if bands_lock[0]:
            return
        bands_lock[0] = True
        if input1.text() == "":
            input1.setText("0")
        if input2.text() == "":
            input2.setText("0")
        if input3.text() == "":
            input3.setText("0")
        if input4.text() == "":
            input4.setText("0")
        bands_lock[0] = False
        bands[region_selector[0]] = (int(input1.text()), int(input2.text()), int(input3.text()), dropdown.currentText(), int(input4.text()))
        update_glyph_selector_win()
        sync_bands()

    ###############################################################################
    # Create label items
    freq_label = pg.LabelItem(justify='top')
    selector_label = pg.LabelItem(justify='bottom')

    # Create input fields restricted to integers
    input1 = QtWidgets.QLineEdit()
    input1.setValidator(QtGui.QIntValidator())  # Allow only integers
    input1.textChanged.connect(update_band)  # Connect dropdown to function

    input2 = QtWidgets.QLineEdit()
    input2.setValidator(QtGui.QIntValidator())  # Allow only integers
    input2.textChanged.connect(update_band)  # Connect dropdown to function

    input3 = QtWidgets.QLineEdit()
    input3.setValidator(QtGui.QIntValidator())  # Allow only integers
    input3.textChanged.connect(update_band)  # Connect dropdown to function

    input4 = QtWidgets.QLineEdit()
    input4.setValidator(QtGui.QIntValidator())  # Allow only integers
    input4.textChanged.connect(update_band)  # Connect dropdown to function

    def toggle_input4():
        if dropdown.currentText() == "tht":
            input4.setEnabled(True)
        else:
            input4.setEnabled(False)
        update_band()

    # Create a dropdown menu
    dropdown = QtWidgets.QComboBox()
    dropdown.addItems(["lin", "tht", "bar"])
    dropdown.currentIndexChanged.connect(toggle_input4)  # Connect dropdown to function

    # Use QGraphicsProxyWidget to embed widgets in pyqtgraph layouts
    input1_proxy = QtWidgets.QGraphicsProxyWidget()
    input1_proxy.setWidget(input1)

    input2_proxy = QtWidgets.QGraphicsProxyWidget()
    input2_proxy.setWidget(input2)

    input3_proxy = QtWidgets.QGraphicsProxyWidget()
    input3_proxy.setWidget(input3)

    dropdown_proxy = QtWidgets.QGraphicsProxyWidget()
    dropdown_proxy.setWidget(dropdown)

    input4_proxy = QtWidgets.QGraphicsProxyWidget()
    input4_proxy.setWidget(input4)

    # Create a layout to hold the labels and input fields
    layout = QtWidgets.QGraphicsGridLayout()
    layout.setContentsMargins(0, 0, 0, 0)  # Remove extra spacing
    layout.setSpacing(2)  # Adjust spacing between elements

    # Add labels to the layout (column 0, row 0 and row 1)
    layout.addItem(freq_label, 0, 0)
    layout.addItem(selector_label, 1, 0)

    # Create a widget container for the input fields and dropdown
    input_layout = QtWidgets.QGraphicsGridLayout()
    input_layout.addItem(input1_proxy, 0, 0)
    input_layout.addItem(input2_proxy, 0, 1)
    input_layout.addItem(input3_proxy, 0, 2)
    input_layout.addItem(dropdown_proxy, 0, 3)
    input_layout.addItem(input4_proxy, 0, 4)

    # Add the input layout below the selector label
    layout.addItem(input_layout, 2, 0)

    # Create a container to hold the layout
    label_container = pg.GraphicsLayout()
    label_container.setLayout(layout)

    win.addItem(label_container)
    bands_lock[0] = True
    toggle_input4()
    bands_lock[0] = False

    # Create a button
    button = QtWidgets.QPushButton("Save")
    button.clicked.connect(update_band)

    # Use QGraphicsProxyWidget to embed the button in pyqtgraph layout
    button_proxy = QtWidgets.QGraphicsProxyWidget()
    button_proxy.setWidget(button)

    # Add the button below the input fields (row 1, column 0, spanning all columns)
    input_layout.addItem(button_proxy, 1, 0, 1, 5)  # Spanning 5 columns to center it
    ###############################################################################

    # Convert frequencies to log scale for selection
    def freq_to_log(freq):
        return np.log10(freq) if freq != 0 else 0

    def log_to_freq(log_val):
        return 10 ** log_val

    # Create a frequency selection region
    region = pg.LinearRegionItem(values=(freq_to_log(20),freq_to_log(277)), movable=True)
    region.setZValue(10)
    plot.addItem(region)

    def repr_leds(bands, idx):
        if len(bands[idx]) < 4:
            return "None"
        leds = ["cam", "diag", "c1", "c2", "c3", "c4", "dot", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
        offset = sum(band[2] for band in bands[:idx])
        return "-".join(map(str, map(int, bands[idx][:2]))) + "(Hz): " + bands[idx][3] + "[" + ", ".join(map(str, leds[offset:offset+bands[idx][2]])) + "]"

    def update_selection(new_selection):
        bands_lock[0] = True
        region_selector[0] = new_selection
        selector_label.setText(f"<span style=\"font-size:{fontsize*0.8}pt;\">Selected item: {repr_leds(bands, new_selection)} @ {new_selection}</span>")
        if (bands[new_selection][0] < bands[new_selection][1]):
            region.setRegion(tuple(map(freq_to_log, bands[new_selection][:2])))
        # Update inputs
        input1.setText(str(int(bands[new_selection][0])))
        input2.setText(str(int(bands[new_selection][1])))
        input3.setText(str(int(bands[new_selection][2])))
        input4.setText(str(int(bands[new_selection][4]) if len(bands[new_selection]) > 4 else ""))
        dropdown.setCurrentText(bands[new_selection][3] if len(bands[new_selection]) > 3 else "")
        bands_lock[0] = False
        update_glyph_selector_win()
    update_selection(0)

    left_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), win)
    right_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), win)
    def al():
        update_selection(region_selector[0] - 1 if region_selector[0] != 0 else len(bands) - 1)
    def ar():
        update_selection(region_selector[0] + 1 if region_selector[0] != len(bands) - 1 else 0)
    left_sc.activated.connect(al)
    right_sc.activated.connect(ar)

    # Function to update frequency label when region selection changes
    def update_selected_range():
        if bands_lock[0]:
            return
        min_freq, max_freq = region.getRegion()
        freq_label.setText(f"<span style=\"font-size:{fontsize}pt;\">Selected Frequency Range: {log_to_freq(min_freq):.2f} Hz - {log_to_freq(max_freq):.2f} Hz</span>")
        bands[region_selector[0]] = (log_to_freq(min_freq),log_to_freq(max_freq)) + bands[region_selector[0]][2:]
        sync_bands()

    region.sigRegionChanged.connect(update_selected_range)
    update_selected_range()  # Initialize label

    # Enable zooming & panning with mouse
    plot.setMouseEnabled(x=True, y=False)

    # Shared index for synchronization
    index = [0]

    # Thread for audio playback
    def play_audio():
        while index[0] < len(fft_results):
            data = wf.readframes(chunk_size)
            if len(data) == 0:
                break
            stream.write(data)
            index[0] += 2  # Move to the next FFT frame

    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    # Function to update the plot
    def update_plot():
        if index[0] < len(fft_results):
            curve.setData(x, fft_results[index[0]])
            min_freq, max_freq = region.getRegion()
            min_freq = log_to_freq(min_freq)
            max_freq = log_to_freq(max_freq)
            #if min_freq >= 0 and max_freq < len(fft_results[index[0]]):
            #    bar_window.update(bass_levels[index[0]])
            glyph_win.glyph_update(glyph_data[0][index[0]])

    # Timer for real-time updates (~60 FPS)
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(int((chunk_size / frame_rate) * 1000))  # Ensure update interval matches audio playback

    # Show UI
    glyph_win.show()
    glyph_win_selector.show()
    win.show()
    app.exec_()

    # Wait for audio to finish
    audio_thread.join()

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
