# Audio Visualizer

<!--- Sample gif and/or image here --->

Given an audio input (.wav file), this project will generate the dynamic visualization shown above as the audio plays.

Once complete, the static visualization may be interacted with in the window according to the PyQtGraph's specified [3D Graphics mouse interactions](http://www.pyqtgraph.org/documentation/mouse_interaction.html#id1).

The final static visualization will be captured and output as an image titled audioVisualization.png. In addition, the amplitude and frequency information gathered per sample read will be output to audioData.csv.

## Getting Started

All library dependencies for this project are available via The Python Package Index.
`pip install numpy scipy`

Note that PyQtGraph requires one of PyQt4, PyQt5 or PySide. We tested with PyQt5.
`pip install PyQt5`

PyQtGraph's PyPI version is not kept up to date. Therefore, to avoid possible resolution issues, install PyQtGraph directly from github as per official recommendation.
`pip install git+https://github.com/pyqtgraph/pyqtgraph`

## Usage

Upload the wav file you would like to visualize into the same directory as the python script.

Change the wav file in the code to be the name of this file. 

<!--- Perhaps change this to be an executable arg? Or a way to just detect the name of the wav file in the folded and use that too? --->

Run the python script:
`python audioMeshCoilVisualizer.py`
