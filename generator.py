import argparse
import base64
import zlib
import subprocess
import os
from mutagen.oggopus import OggOpus
import threading
import threading
from fft import precompute_fft, glyph_compute

# Validate OGG file to ensure it has a valid Opus stream
def validate_ogg(ogg_file):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", ogg_file], capture_output=True, text=True)
        if "codec_name=opus" not in result.stdout:
            raise ValueError("Invalid Ogg Opus file format.")
    except Exception as e:
        raise RuntimeError(f"FFmpeg validation failed: {e}")

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
