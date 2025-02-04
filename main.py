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

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

# Precompute FFT for the entire WAV file
def precompute_fft(wav_file, chunk_size=8192):
    wf = wave.open(wav_file, 'rb')
    frame_rate = wf.getframerate()
    total_frames = wf.getnframes()
    fft_results = []
    timestamps = []
    
    for i in range(0, total_frames, chunk_size):
        data = wf.readframes(chunk_size)
        if len(data) < chunk_size * wf.getsampwidth():
            break
        audio_data = np.frombuffer(data, dtype=np.int16)
        fft_data = np.abs(np.fft.fft(audio_data)[:chunk_size//2])
        fft_results.append(fft_data)
        timestamps.append(i / frame_rate)

    wf.close()
    return fft_results, timestamps, frame_rate, chunk_size

# Validate OGG file to ensure it has a valid Opus stream
def validate_ogg(ogg_file):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", ogg_file], capture_output=True, text=True)
        if "codec_name=opus" not in result.stdout:
            raise ValueError("Invalid Ogg Opus file format.")
    except Exception as e:
        raise RuntimeError(f"FFmpeg validation failed: {e}")

# Generate an OGG file compatible with Nothing Glyph
def generate_nothing_ogg(wav_file, output_file, low_freq, high_freq, inspect=False):
    fft_results, timestamps, frame_rate, chunk_size = precompute_fft(wav_file, chunk_size=8192)
    bass_levels = []

    # Extract frequencies in the specified range
    low_index = int(low_freq / (frame_rate / chunk_size))
    high_index = int(high_freq / (frame_rate / chunk_size))

    for fft_data in fft_results:
        band_amplitude = np.sum(fft_data[low_index:high_index])
        bass_levels.append(int((band_amplitude / max(1, np.max(fft_data))) * 4095))

    # Remap values to between 0 and 4095 (max value for each led)
    intr = interp1d([0,max(bass_levels)], [0,4095])
    bass_levels = np.asarray(np.array(intr(bass_levels)), dtype="int")
    
    # Generate CSV light data for USB Line
    csv_lines = []
    for level in bass_levels:
        csv_lines.append(f"0,0,0,0,0,0,0,{level},{level},{level},{level},{level},{level},{level},{level},\r\n")
    
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

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process MP3 and generate Nothing-compatible OGG files or visualize audio.")
    parser.add_argument("-c", action="store_true", help="Create an OGG file for Nothing Glyph Composer")
    parser.add_argument("-i", action="store_true", help="Inspect CUSTOM1 data when used with -c")
    parser.add_argument("-l", type=int, default=400, help="Lower frequency bound for bass extraction (Hz)")
    parser.add_argument("-u", type=int, default=700, help="Upper frequency bound for bass extraction (Hz)")
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
