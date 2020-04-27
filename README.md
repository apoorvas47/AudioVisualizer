# AudioSquare

<!--- Sample gif and/or image here --->

Given an audio .wav file input (with a 44.1 kHz sampling rate for best results), this project will generate the dynamic visualization shown above as the audio plays. Once complete, the static visualization may be interacted with in the window according to the PyQtGraph's specified [3D Graphics mouse interactions](http://www.pyqtgraph.org/documentation/mouse_interaction.html#id1). The final static visualization will be captured and output as an image titled audioVisualization.png. In addition, the amplitude and frequency information gathered per sample read will be output to audioData.csv.

## Getting Started

All library dependencies for this project are available via The Python Package Index.
`pip install numpy scipy`

Note that PyQtGraph requires one of PyQt4, PyQt5 or PySide. We tested with PyQt5.
`pip install PyQt5`

PyQtGraph's PyPI version is not kept up to date. Therefore, to avoid possible resolution issues, install PyQtGraph directly from github as per official recommendation.
`pip install git+https://github.com/pyqtgraph/pyqtgraph`

## Usage

Upload the .wav file you would like to visualize into the same directory as the python script.

If there are multiple .wav files in the directory, specify which to use as the second argument when running the python script. 

Run the python script:
`python audioMeshCoilVisualizer.py <wav file name>`
