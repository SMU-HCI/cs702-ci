import librosa
import numpy as np
import pyqtgraph as pg
import pyaudio
import joblib
from pyqtgraph.Qt import QtCore, QtWidgets
from threading import Thread
import matplotlib.cm as cm
import matplotlib.colors as colors

FS = 22050  # Hz
CHUNK_SIZE = 512  # samples


class MicrophoneRecorder(QtCore.QObject):
    signal = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=FS,
                                  input=True,
                                  frames_per_buffer=CHUNK_SIZE)
        self.running = True

    def run(self):
        while self.running:
            data = self.stream.read(CHUNK_SIZE)
            y = np.frombuffer(data, 'int16')
            y = y.astype(float)
            self.signal.emit(y)

    def close(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class SpectrogramWidget(pg.PlotWidget):
    def __init__(self):
        super(SpectrogramWidget, self).__init__()
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.label = pg.LabelItem(justify='right')
        self.addItem(self.label)
        self.label.setPos(0, 0)
        self.label.setText("Detected: ")
        self.invertY()

        self.ring_buffer_size = 256
        self.feature_size = 128
        self.feature_frame_length = 44

        self.img_array = np.zeros((self.ring_buffer_size, self.feature_size))

        self.model = joblib.load("model.pkl")
        self.counter = 0

        # Create a colormap similar to the image
        colormap = cm.get_cmap('inferno')
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        lut = lut[:-3]  # Remove the last few colors to make it darker at the top

        self.img.setLookupTable(lut)
        self.img.setLevels([-80, 0])  # Adjust the levels to make it darker
        self.img.scale()

        self.setLabel('left', 'Frequency', units='Hz')
        self.setLabel('bottom', 'Time', units='frames')

        self.win = np.hanning(CHUNK_SIZE)
        self.show()

    def update(self, chunk):
        # Convert raw sound data to melspectrogram
        S = librosa.feature.melspectrogram(y=chunk, sr=FS, n_mels=self.feature_size, fmax=8000, hop_length=256,
                                           n_fft=512)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Roll the img_array to the left by one frame
        self.img_array = np.roll(self.img_array, -1, axis=0)
        # Update the rightmost frame with the new melspectrogram frame
        self.img_array[-1, :] = S_db[:, 0]

        # Perform detection
        features = self.img_array[-self.feature_frame_length:, :]

        # The feature dimension should be (1, 5632)
        features = features.reshape(1, -1)
        label = self.model.predict(features)

        if label == 0:
            self.label.setText("Detected: Scratch")
            self.counter = self.feature_frame_length  # Keep the label alive for a few frames
        else:
            self.counter -= 1

        if self.counter <= 0:
            self.label.setText("Detected: None")

        self.img.setImage(self.img_array, autoLevels=False)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    w = SpectrogramWidget()

    mic = MicrophoneRecorder()
    mic.signal.connect(w.update)

    thread = Thread(target=mic.run)
    thread.start()

    app.exec()
    mic.close()
    thread.join()
