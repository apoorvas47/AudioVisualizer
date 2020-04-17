"""
Dynamic coiling audio visualizer for wav files using pyaudio and pyqtgraph
"""

import numpy as np
from numpy.fft import rfft
from pyqtgraph import opengl
from pyqtgraph.Qt import QtCore, QtGui
import struct
from pyaudio import PyAudio
import sys
import wave
import math

class AudioVisualizer(object):
    def __init__(self):
        """
        Add the empty mesh surface to the graphics view widget window.
        """

        # Must construct and save a QApplication before a QWidget.
        self.app = QtGui.QApplication(sys.argv)

        self.window = opengl.GLViewWidget()

        # *** REPLACE wavefile.wav WITH THE WAV FILE NAME IN THIS FOLDER ***
        # For best results, use audio files with the standard sample rate
        #   of 44100Hz (Can verify this via self.wf.getframerate())
        self.waveform = wave.open("shortSample.wav", 'rb')

        total_frames = self.waveform.getnframes()
        self.frame_rate = self.waveform.getframerate()
        self.frames_per_sample = math.floor(self.frame_rate / 8.74)
        # 8.74 just a num I found that works well

        # This is the width used for the squares made for each sample wf taken.
        # This number is 3 by default but can be altered as desired to a
        #   reasonable value greater than 1. (Recommended maximum of 10)
        self.square_width = 3

        # Initialize audio data file with header row of corresponding labels
        with open('audioData.csv', "w") as csv_file:
            amplitudes_per_sample = self.square_width**2
            labels = ['amplitude %s'%i for i in range(1, amplitudes_per_sample +1)]
            labels.append('frequency\n')
            csv_file.write(', '.join(labels))

        self.total_squares = math.ceil(total_frames / self.frames_per_sample)
        self.grid_width = math.ceil(math.sqrt(self.total_squares)) + \
                          2 * self.square_width
        center_coord = math.ceil(self.grid_width / 2)
        self.current_coord = next_coil_coordinate(center_coord,
                                                  center_coord,
                                                  self.total_squares)
        self.squares_so_far = 0
        self.grid_width += self.square_width
        # Necessary since current coord will give the corner coordinate

        # Note: the below are just tried and tested constants
        # Should I explain what each of these mean?
        self.cam_distance = 5
        self.cam_azimuth = 230 # Varies by % 360
        self.cam_elevation = 20 # Varies from -90 to 90

        # Kept track of for camera position measurements
        self.totalOrbitsCount = 0

        self.window.setCameraPosition(distance=self.cam_distance,
                                      elevation=self.cam_elevation,
                                      azimuth=self.cam_azimuth)

        self.ypoints = np.arange(- self.grid_width / 2, self.grid_width / 2, 1)
        self.xpoints = np.arange(- self.grid_width / 2, self.grid_width / 2, 1)
        # Note the length of these points arrays should be self.grid_width

        # Create stream to play audio on.
        audio_format = PyAudio().get_format_from_width(
            self.waveform.getsampwidth())
        self.stream = PyAudio().open(format= audio_format,
                                     channels=self.waveform.getnchannels(),
                                     rate=self.frame_rate,
                                     output=True)

        self.update_mesh_args(initialized=False)

        self.mesh_item = opengl.GLMeshItem(
            faces=self.faces,
            vertexes=self.verts,
            faceColors=self.colors,
            drawEdges=True,
            smooth=False,
            edgeColor=(0,0,0,0)
        )

        self.mesh_item.setGLOptions('additive')
        self.window.addItem(self.mesh_item)
        self.window.setWindowTitle('Audio Mesh Visualizer')
        self.window.setFixedHeight(700)
        self.window.setFixedWidth(700)
        self.window.show()

    def update_mesh_args(self, wf_sample=None, initialized=True):
        """
        Update mesh verts, faces, and colors based on data from the waveform sample.
        TODO: Update description if necessary
        """

        # Calculate data
        if wf_sample is not None:
            wf_len = len(wf_sample)
            # Note: length of waveform_data should initially be 4 times the
            # number of frames read

            wf_sample = struct.unpack(str(wf_len) + 'B', wf_sample)

            if wf_len == 0: # TODO: see if this additional check can be refact
                self.timer.stop()
                self.window.grabFrameBuffer().save('audioVisualization.png')
                return

            # Calculate color to use from sample

            rfft_data = np.abs(rfft(wf_sample))
            # Note that the length of rfft_data should be equivalent to
            # initial waveform length / 2 and then + 1 for the DC term
            rfft_sum = 0
            sum_of_arr = 0
            rfft_len = len(rfft_data)
            freq_factor = self.frame_rate / 2 / rfft_len
            for i in range(1, rfft_len):
                freqAtIndex = i * freq_factor
                rfft_sum += rfft_data[i] * freqAtIndex
                sum_of_arr += rfft_data[i]
            avg_freq = 0 if sum_of_arr == 0 else rfft_sum / sum_of_arr
            square_color = self.color_from_freq(avg_freq)

            # Retreive square width ^ 2 points of amplitude to use.

            totalStepsNeeded = self.square_width**2 - 1
            step = math.floor(len(wf_sample) / totalStepsNeeded)
            # To prevent trimming sample one too short...
            # (occurs when sample (frames read * 4) % 8 == 0 ie frames read
            # is odd)
            if len(wf_sample) % step == 0:
                step -= 1
            wf_sample = np.array(wf_sample, dtype='b')[::step]
            # If data too short to be reshaped to appropriate square size
            # (square width ^ 2) anyway...
            if len(wf_sample) < self.square_width**2:
                self.timer.stop()
                self.window.grabFrameBuffer().save('audioVisualization.png')
                return

            wf_sample = wf_sample * 0.03 # Factor here affects height seen

            # Write amplitudes and avg frequency to audio data csv file
            with open('audioData.csv', "a") as csv_file:
                csv_file.write(', '.join([str(ampl) for ampl in wf_sample]))
                csv_file.write(', ' + str(avg_freq))
                csv_file.write('\n')

            wf_sample = wf_sample.reshape((self.square_width, self.square_width))
        else:
            wf_sample = np.array([0] * self.square_width * self.square_width)
            wf_sample = wf_sample.reshape((self.square_width, self.square_width))
            square_color = [0, 0, 0, 0]
            # note: each color is [R,G,B,Translucency]

        # Set data
        if not initialized:
            # Initialze the grid with its default verts, faces, and colors.

            self.verts = np.array([
                [
                    x, y, 0
                ] for xid, x in enumerate(self.xpoints) for yid, y in
                enumerate(self.ypoints)
            ], dtype=np.float32)

            faces = []
            colors = []
            for yid in range(self.grid_width - 1):
                yoff = yid * self.grid_width
                for xid in range(self.grid_width - 1):
                    faces.append([
                        xid + yoff,
                        xid + yoff + self.grid_width,
                        xid + yoff + self.grid_width + 1,
                    ])
                    faces.append([
                        xid + yoff,
                        xid + yoff + 1,
                        xid + yoff + self.grid_width + 1,
                    ])
                    colors.append([0, 0, 0, 1])
                    colors.append([0, 0, 0, 1])

            self.faces = np.array(faces, dtype=np.uint32)
            self.colors = np.array(colors, dtype=np.float32)
        else:
            # Just update the verts the colors of corresponding sample
            # square now.

            xcoord, ycoord, direc = next(self.current_coord)
            self.squares_so_far += 1

            for yid in range(self.square_width):
                yoff = yid + ycoord
                for xid in range(self.square_width):
                    xoff = xid + xcoord

                    xpush = 0
                    ypush = 0
                    if direc==2:
                        ypush = 1
                    if direc==1:
                        xpush = 1

                    vert_coord = int((yoff+ypush) * self.grid_width + (xoff + xpush))
                    if vert_coord >= self.grid_width**2:
                        break

                    # update the height or z-axis value of coordinate
                    self.verts[vert_coord][2] = wf_sample[yid % 3][xid]

                    color_coord = ((xoff) - (yoff))*2 + (
                            yoff * self.grid_width * 2)
                    # e.g. (4,6) --> (4 - 5) * 2 + 6 * 12 * 2

                    # Update the color of both triangle faces of the square:

                    if color_coord >= len(self.colors):
                        continue
                    self.colors[color_coord] = square_color

                    color_coord += 1
                    if color_coord >= len(self.colors):
                        continue
                    self.colors[color_coord] = square_color

    def update(self):
        """
        Read and play audio and update gl mesh item and window camera.
        """

        # Read self.frames_per_square frames.
        waveform_sample = self.waveform.readframes(self.frames_per_sample)

        # Play audio
        self.stream.write(waveform_sample)

        # Update mesh item

        self.update_mesh_args(wf_sample=waveform_sample)
        self.mesh_item.setMeshData(vertexes=self.verts, faces=self.faces,
                                   faceColors=self.colors)

        # Update camera movement

        exponentialGrowthFactor = .75

        self.cam_azimuth -= 90 / (self.totalOrbitsCount * 2 + self.square_width)

        self.cam_distance += .25 - .23 * (
                    1 - exponentialGrowthFactor ** self.totalOrbitsCount)

        # (20 was the initial cam position)
        self.cam_elevation = 20 + (90 - 20) * self.squares_so_far / \
                             self.total_squares

        self.window.setCameraPosition(azimuth=self.cam_azimuth,
                                      distance=self.cam_distance,
                                      elevation=self.cam_elevation)

        if abs(self.cam_azimuth / 360) > self.totalOrbitsCount:
            self.totalOrbitsCount += 1

    def color_from_freq(self, freq):
        """
        Returns an RGB color value array to use from a frequency.
        """

        max_freq = self.frame_rate / 2

        # After sampling many songs, this were the constants that captured
        # the range of most
        range_lower_bound = max_freq / 2.15
        range_upper_bound = max_freq / 1.95
        range = range_upper_bound - range_lower_bound

        rescaled_freq = freq - range_lower_bound
        if rescaled_freq < 0:
            rescaled_freq = 0
        elif rescaled_freq > range:
            rescaled_freq = 1

        # Converts the frequency to a value between 0 and 1.
        # A lower frequency will produce a bluer value and a higher one will
        #   produce a redder value.
        freq_val = rescaled_freq / range
        return [freq_val, .3, 1 - freq_val, 1]

    def start(self):
        """
        Begins the timer and sets up the graphics window.
        """
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


def next_coil_coordinate(x, y, total_squares):
    """
    Yields the next coordinate and direction being travelled each time.
    """

    total = 0

    coil_squares_length = 1
    coil_width = 1
    current_x = x
    current_y = y
    direction = 1 # Default.

    yield (current_x, current_y, direction) # Yield first point first.
    coil_completed_squares = 1
    total += 1

    while total < total_squares:
        total += 1

        # Completed coil loop, at the bottom left corner.
        # Must go down and begin a new coil loop.
        if coil_completed_squares == coil_squares_length:
            current_x += 1
            coil_width += 2
            coil_squares_length = coil_width*4 - 4
            coil_completed_squares = 1
            progress_to_next_corner = 2
            direction = 1
            yield (current_x, current_y, direction)
            continue

        # move to next position:
        if direction == 1:
            current_y += 1
            coil_completed_squares += 1
            progress_to_next_corner += 1
        elif direction == 2:
            current_x -= 1
            coil_completed_squares += 1
            progress_to_next_corner += 1
        elif direction == 3:
            current_y -= 1
            coil_completed_squares += 1
            progress_to_next_corner += 1
        elif direction == 4:
            current_x += 1
            coil_completed_squares += 1
            progress_to_next_corner += 1

        # if you were at a corner, change direction too!
        if progress_to_next_corner == coil_width:
            direction += 1
            progress_to_next_corner = 1
            # At direction = 4 increment you should be making a new coil anyway

        yield (current_x, current_y, direction)


if __name__ == '__main__':
    visualizer = AudioVisualizer()
    visualizer.start()
