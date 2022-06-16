import tensorflow.keras as keras
from tqdm import tqdm as tqdm

class TQDMPredictCallback(keras.callbacks.Callback):
    def __init__(self, total, **tqdm_params):
        super().__init__()
        self.tqdm_progress = None
        self.prev_predict_batch = None
        self.total = total
        self.tqdm_params = tqdm_params

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.tqdm_progress.update(batch - self.prev_predict_batch)
        self.prev_predict_batch = batch

    def on_predict_begin(self, logs=None):
        self.prev_predict_batch = 0
        total = self.total
        if total:
            total -= 1

        self.tqdm_progress = tqdm(total=self.total, **self.tqdm_params)

    def on_predict_end(self, logs=None):
        if self.tqdm_progress:
            self.tqdm_progress.close()