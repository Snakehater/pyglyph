# Audio Visualizer and Nothing Glyph Composer OGG Generator

This project processes MP3 files, converts them to WAV, and either visualizes their frequency spectrum or generates an OGG file compatible with the Nothing Glyph Composer.

## Features
- Converts MP3 to WAV for processing.
- Visualizes the frequency spectrum of an audio file.
- Generates an OGG file with embedded metadata for Nothing Glyph Composer.
- Supports frequency band customization.

## Installation

Ensure you have all required dependencies installed by running:

```bash
pip install -r requirements.txt
```

## Usage

Run the script using:

```bash
python main.py [-c] [-i] [-b BANDS_FILE] filename.mp3
```

### Arguments
- `filename.mp3`: The input MP3 file to process.
- `-c`: Generate an OGG file compatible with the Nothing Glyph Composer.
- `-i`: Inspect the CUSTOM1 data when used with `-c`.
- `-b BANDS_FILE`: (Optional) Specify a bands configuration file. Default bands file is "custom_bands.bands"

### Example Commands

**Visualizing an audio file:**
```bash
python main.py my_audio.mp3
```

**Generating a Nothing Glyph-compatible OGG file:**
```bash
python main.py -c my_audio.mp3
```

**Using a custom band configuration for visualization or OGG generation:**
```bash
python main.py -b custom_bands.bands my_audio.mp3
```

## File Descriptions

- `main.py`: Entry point of the program. Handles argument parsing and manages visualization or OGG generation.
- `generator.py`: Handles the conversion of audio into Nothing Glyph-compatible OGG files with metadata embedding.
- `visualizer.py`: Provides real-time visualization of audio frequency bands using PyQtGraph.
- `fft.py`: Computes the FFT (Fast Fourier Transform) for audio frequency analysis.

## Notes
- The visualization supports key bindings:
  - `Q`: Quit the session.
  - `Space`: Pause/resume playback.
  - `Left/Right Arrow`: Navigate between frequency bands.
- The generated OGG files contain embedded frequency-based light sequences for Nothing devices.
```

## TODO
- Make the user have more control over the bands configuration by adding new bands or removing unused bands, now the user have to manually configure the number of bands by modifying the bands file
