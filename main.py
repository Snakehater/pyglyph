import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
import base64
import zlib
import subprocess
import os
from pydub import AudioSegment
from mutagen.oggopus import OggOpus
import threading
import time
from scipy.interpolate import interp1d
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import threading
from glyph import GlyphUI

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

# Precompute FFT for the entire WAV file
def precompute_fft(wav_file, chunk_size=2048, overlap=0.5):
    wf = wave.open(wav_file, 'rb')
    frame_rate = wf.getframerate()
    total_frames = wf.getnframes()
    step_size = int(chunk_size * (1 - overlap)) # Overlapping windows for more fft data
    fft_results = []
    timestamps = []
    
    for i in range(0, total_frames - chunk_size, step_size):
        wf.setpos(i)
        data = wf.readframes(chunk_size)
        if len(data) < chunk_size * wf.getsampwidth():
            break
        audio_data = np.frombuffer(data, dtype=np.int16)
        fft_data = np.abs(np.fft.fft(audio_data)[:chunk_size//2])
        fft_results.append(fft_data)
        timestamps.append(i / frame_rate)

    wf.close()
    print(f"FFT Computation: Frequency range: 0Hz to {frame_rate/2:.2f}Hz, Resolution: {len(fft_results[0])} Chunk duration: {chunk_size/frame_rate:.3f} seconds, Data length: {(chunk_size/frame_rate)*len(fft_results):.3f}")
    return fft_results, timestamps, frame_rate, chunk_size

# Validate OGG file to ensure it has a valid Opus stream
def validate_ogg(ogg_file):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", ogg_file], capture_output=True, text=True)
        if "codec_name=opus" not in result.stdout:
            raise ValueError("Invalid Ogg Opus file format.")
    except Exception as e:
        raise RuntimeError(f"FFmpeg validation failed: {e}")

def bar(level, entries, genarr=False):
    ret = ""
    retarr = []
    span = 4095/entries
    for i in range(entries):
        if i != 0 and not genarr:
            ret += ","
        if level > span*(i + 1):
            if genarr:
                retarr += [255]
            else:
                ret += "4095"
        elif genarr:
            retarr += [0]
        else:
            ret += "0"
    if genarr:
        return retarr
    return ret

def bands_levels_gen(fft_results, bands):
    bands_levels = []
    max_levels = []
    # Make sure we have the correct amount of bands in the bands_levels
    for band in bands:
        if band[0] == band[1]:
            bands_levels.append(None)
            max_levels.append(0)
        else:
            bands_levels.append([])
            ## TODO remove the one that does not work, np is more efficiant if the conversion to np array is fast enough
            max_levels.append(np.max(np.array(fft_results)[:, band[0]:band[1]]))
            #max_levels.append(max(max(x[band[0]:band[1]]) for x in fft_results))

    # iterate over fft_results and for each frame, add maximum value in each band to each band
    for fft_frame in fft_results:
        for i in range(len(bands)):
            if bands_levels[i] is not None:
                bands_levels[i].append(int((np.max(fft_frame[bands[i][0]:bands[i][1]]) / max_levels[i]) * 4095))

    return bands_levels

def band_compute(bands, fft_results, timestamps, frame_rate, chunk_size, simulation):
    for i in range(len(bands)):
        # Extract frequencies in the specified range
        low_index = int(bands[i][0] / (frame_rate / chunk_size))
        high_index = int(bands[i][1] / (frame_rate / chunk_size))
        bands[i] = (low_index, high_index, bands[i][2])

    # Generate levels
    bands_levels = bands_levels_gen(fft_results, bands)

    if not simulation:
        expected_frames = int((timestamps[-1] - timestamps[0]) * 60)  # Total number of frames needed
        for i in range(len(bands_levels)):
            if bands_levels[i] is None:
                continue
            # Ensure each frame corresponds to 1/60th of a second
            # and ensure smooth brightness updates per 1/60s
            levels = bands_levels[i]
            levels = np.interp(np.linspace(0, len(levels)-1, expected_frames), np.arange(len(levels)), levels)
            levels = np.round(levels).astype(int)
            bands_levels[i] = levels

    max_length = 0
    if not all(x is None for x in bands_levels):
        max_length = max(len(x) for x in bands_levels if x is not None) # bands_levels and max length of an arr
    return bands_levels, max_length

def glyph_compute(bands, fft_results, timestamps, frame_rate, chunk_size, simulation=False):
    ret = []
    # Will return an array of exactly the same numbers of bands, even if they are not used, the not used ones will have None as element, the others will have an array of the levels
    bands_levels, length = band_compute(bands.copy(), fft_results, timestamps, frame_rate, chunk_size, simulation)

    # Combine levels from each band and put them into the correct led section of the glyph data
    # Iterate over all levels
    for i in range(length):
        glyph_data = []
        # For each level on each band, add them to the correct glyph index
        # Iterate over all bands and push them to the correct element in the glyph data
        for b in range(len(bands)):
            # Band not used
            if bands_levels[b] is None:
                # Fill unused leds with zeroes, number of leds covered by band is third index:
                for j in range(bands[b][2]):
                    glyph_data.append(0)
            # Band used
            else:
                match bands[b][3]:
                    case "bar":
                        glyph_data += bar(bands_levels[b][i], bands[b][2], genarr=True)
                    case "lin":
                        v = bands_levels[b][i] if not simulation else int((bands_levels[b][i]/4095)*255)
                        for j in range(bands[b][2]):
                            glyph_data.append(v)
                    case "tht":
                        v = (4095 if not simulation else 255) if bands_levels[b][i] > bands[b][4] else 0
                        for j in range(bands[b][2]):
                            glyph_data.append(v)
        if simulation:
            ret.append(glyph_data)
        else:
            ret.append(",".join(map(str, glyph_data)) + ",\r\n")
        glyph_data = []
    return ret

# Generate an OGG file compatible with Nothing Glyph
def generate_nothing_ogg(wav_file, output_file, inspect=False):
    fft_results, timestamps, frame_rate, chunk_size = precompute_fft(wav_file, chunk_size=2048*4, overlap=0.5)

    bands = [
        (462,2753,1,"lin"),
        (462,2753,1,"lin"),
        (20,277,4,"tht", 1024*3),
        (20,277,9, "bar")
    ]

    # Generate CSV light data for USB Line
    csv_lines = glyph_compute(bands, fft_results, timestamps, frame_rate, chunk_size)

    # Add blank line at the end to turn off the glyph leds
    csv_lines.append(f"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\r\n")
    
    # Compress and encode `AUTHOR` tag
    author_data = base64.b64encode(zlib.compress("".join(csv_lines).encode("utf-8"), level=9)).decode("utf-8")
    custom1_data = ",".join([f"{int(t * 1000)}-3" for t in timestamps]) + ","
    
    # Convert WAV to OGG using FFmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_file, "-c:a", "libopus", "-b:a", "96k", "-strict", "-2", "-application", "voip", "-f", "ogg", output_file
    ], check=True)

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError("FFmpeg failed to produce a valid OGG Opus file.")
    
    validate_ogg(output_file)
    
    # Add metadata using mutagen
    ogg = OggOpus(output_file)
    ogg.tags.update({
        "TITLE": "Generated Composition",
        "ALBUM": "Custom Sound Pack",
        "AUTHOR": author_data,
        "CUSTOM1": base64.b64encode(zlib.compress(custom1_data.encode("utf-8"), level=9)).decode("utf-8"),
        "CUSTOM2": "5cols",
        "COMPOSER": "v1-Spacewar Glyph Composer"
    })
    ogg.save()
    print(f"✅ OGG file saved: {output_file}")
    
    if inspect:
        print("Decoded csv_lines:")
        print("".join(csv_lines))
        print(f"Number of lines: {len(csv_lines)}")
        print(f"Sound length: {ogg.info.length}s")
        print(f"Glyph length: {len(csv_lines)*(1/60)}s")

def audio_visualizer(wav_file):
    print("Running optimized audio visualizer...")
    
    # Precompute FFT for visualization
    fft_results, timestamps, frame_rate, chunk_size = precompute_fft(wav_file, chunk_size=2048*1, overlap=0.5)

    region_selector = [0]
    bands_lock = [False]

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
        region_selector[0] = new_selection
        selector_label.setText(f"Selected item: {repr_leds(bands, new_selection)} @ {new_selection}")
        if (bands[new_selection][0] < bands[new_selection][1]):
            region.setRegion(tuple(map(freq_to_log, bands[new_selection][:2])))
        # Update inputs
        bands_lock[0] = True
        input1.setText(str(int(bands[new_selection][0])))
        input2.setText(str(int(bands[new_selection][1])))
        input3.setText(str(int(bands[new_selection][2])))
        input4.setText(str(int(bands[new_selection][4]) if len(bands[new_selection]) > 4 else ""))
        dropdown.setCurrentText(bands[new_selection][3] if len(bands[new_selection]) > 3 else "")
        bands_lock[0] = False
        update_glyph_selector_win()
    update_selection(0)

    up_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), win)
    down_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), win)
    def au():
        update_selection(region_selector[0] - 1 if region_selector[0] != 0 else len(bands) - 1)
    def ad():
        update_selection(region_selector[0] + 1 if region_selector[0] != len(bands) - 1 else 0)
    up_sc.activated.connect(au)
    down_sc.activated.connect(ad)

    # Function to update frequency label when region selection changes
    def update_selected_range():
        min_freq, max_freq = region.getRegion()
        freq_label.setText(f"Selected Frequency Range: {log_to_freq(min_freq):.2f} Hz - {log_to_freq(max_freq):.2f} Hz")
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

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process MP3 and generate Nothing-compatible OGG files or visualize audio.")
    parser.add_argument("-c", action="store_true", help="Create an OGG file for Nothing Glyph Composer")
    parser.add_argument("-i", action="store_true", help="Inspect CUSTOM1 data when used with -c")
    args = parser.parse_args()
    
    mp3_file = "pedro.mp3"
    wav_file = "temp.wav"
    output_file = "pedro_nothing.ogg"
    
    convert_mp3_to_wav(mp3_file, wav_file)
    
    if args.c:
        generate_nothing_ogg(wav_file, output_file, inspect=args.i)
    else:
        audio_visualizer(wav_file)

if __name__ == "__main__":
    main()
