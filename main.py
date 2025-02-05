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

def bar(level, entries):
    ret = ""
    span = 4095/entries
    for i in range(entries):
        if i != 0:
            ret += ","
        if level > span*(i + 1):
            ret += "4095"
        else:
            ret += "0"
    return ret

def bass_levels_gen_idx(fft_results, low_index, high_index):
    bass_levels = []
    max_level = max(max(x) for x in fft_results)
    for fft_data in fft_results:
        #band_amplitude = np.sum(fft_data[low_index:high_index])
        #bass_levels.append(int((band_amplitude / max(1, np.max(fft_data))) * 4095))
        bass_levels.append(int(np.max(fft_data[low_index:high_index] / max_level) * 4095))

    # Remap values to between 0 and 4095 (max value for each led)
    intr = interp1d([0, max(bass_levels)], [0, 4095], fill_value=(0, 4095), bounds_error=False)
    bass_levels = np.asarray(np.array(intr(bass_levels)), dtype="int")
    return bass_levels, max_level

def bass_levels_gen(fft_results, frame_rate, chunk_size, low_freq, high_freq):
    # Extract frequencies in the specified range
    low_index = int(low_freq / (frame_rate / chunk_size))
    high_index = int(high_freq / (frame_rate / chunk_size))
    return bass_levels_gen_idx(fft_results, low_index, high_index)

# Generate an OGG file compatible with Nothing Glyph
def generate_nothing_ogg(wav_file, output_file, low_freq, high_freq, inspect=False):
    fft_results, timestamps, frame_rate, chunk_size = precompute_fft(wav_file, chunk_size=2048*4, overlap=0.5)
    bass_levels = []

    # Extract frequencies in the specified range
    low_index = int(low_freq / (frame_rate / chunk_size))
    high_index = int(high_freq / (frame_rate / chunk_size))

    # Generate bass levels
    bass_levels, max_level = bass_levels_gen_idx(fft_results, low_index, high_index)
    
    # Ensure each frame corresponds to 1/60th of a second
    # and ensure smooth brightness updates per 1/60s
    expected_frames = int((timestamps[-1] - timestamps[0]) * 60)  # Total number of frames needed
    bass_levels = np.interp(np.linspace(0, len(bass_levels)-1, expected_frames), np.arange(len(bass_levels)), bass_levels)
    bass_levels = np.round(bass_levels).astype(int)
    
    # Generate CSV light data for USB Line
    csv_lines = []
    for level in bass_levels:
        csv_lines.append(f"0,0,0,0,0,0,0,{bar(level, 8)},\r\n")
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
        print(f"Number of lines: {len(bass_levels)}")
        print(f"Sound length: {ogg.info.length}s")
        print(f"Glyph length: {len(bass_levels)*(1/60)}s")

class BarWindow(QtWidgets.QWidget):
    """ A separate window with eight vertically stacked colored boxes. """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bar")
        self.setGeometry(300, 300, 300, 800)  # Adjusted height for 8 boxes
        layout = QtWidgets.QVBoxLayout()

        # Default colors for the boxes
        self.boxes = []

        for i in range(8):
            box = QtWidgets.QLabel()
            box.setStyleSheet(f"background-color: black; min-height: 80px;")
            layout.addWidget(box)
            self.boxes.append(box)

        self.setLayout(layout)
        self.show()  # Show the window immediately

    def update(self, level):
        entries = 8
        span = 4095/entries
        for i in range(entries):
            if level > span*(i + 1):
                self.boxes[entries - 1 - i].setStyleSheet(f"background-color: white; min-height: 80px")
            else:
                self.boxes[entries - 1 - i].setStyleSheet(f"background-color: black; min-height: 80px")

def audio_visualizer(wav_file):
    print("Running optimized audio visualizer...")
    
    # Precompute FFT for visualization
    fft_results, timestamps, frame_rate, chunk_size = precompute_fft(wav_file, chunk_size=2048*1, overlap=0.5)
    bass_levels, max_level = bass_levels_gen(fft_results, frame_rate, chunk_size, 0, 100)

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

    # Label to display selected frequency range
    freq_label = pg.LabelItem(justify='right')
    win.addItem(freq_label)

    # Convert frequencies to log scale for selection
    def freq_to_log(freq):
        return np.log10(freq)

    def log_to_freq(log_val):
        return 10 ** log_val

    # Create a frequency selection region
    region = pg.LinearRegionItem(values=(freq_to_log(20),freq_to_log(277)), movable=True)
    region.setZValue(10)
    plot.addItem(region)

    # Function to update frequency label when region selection changes
    def update_selected_range():
        min_freq, max_freq = region.getRegion()
        freq_label.setText(f"Selected Frequency Range: {log_to_freq(min_freq):.2f} Hz - {log_to_freq(max_freq):.2f} Hz")
        bass_levels, max_level = bass_levels_gen(fft_results, frame_rate, chunk_size, log_to_freq(min_freq), log_to_freq(max_freq))

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

    # Show the bar window
    bar_window = BarWindow()

    # Function to update the plot
    def update_plot():
        if index[0] < len(fft_results):
            curve.setData(x, fft_results[index[0]])
            min_freq, max_freq = region.getRegion()
            min_freq = log_to_freq(min_freq)
            max_freq = log_to_freq(max_freq)
            if min_freq >= 0 and max_freq < len(fft_results[index[0]]):
                bar_window.update(bass_levels[index[0]])

    # Timer for real-time updates (~60 FPS)
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(int((chunk_size / frame_rate) * 1000))  # Ensure update interval matches audio playback

    # Show UI
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
    parser.add_argument("-l", type=int, default=20, help="Lower frequency bound for bass extraction (Hz)")
    parser.add_argument("-u", type=int, default=210, help="Upper frequency bound for bass extraction (Hz)")
    args = parser.parse_args()
    
    mp3_file = "pedro.mp3"
    wav_file = "temp.wav"
    output_file = "pedro_nothing.ogg"
    
    convert_mp3_to_wav(mp3_file, wav_file)
    
    if args.c:
        generate_nothing_ogg(wav_file, output_file, args.l, args.u, inspect=args.i)
    else:
        audio_visualizer(wav_file)

if __name__ == "__main__":
    main()
