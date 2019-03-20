import numpy as np
import random
from utils import padtomaxlen
import tensorflow as tf
import threading


def load_metadata(meta_path, split=1):
    def item(line):
        return line.split('|', 3)[:2]  # audo,mel path

    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = [item(line) for line in f]

    if split != 1:
        train_data_len = int(len(metas) * split)
        return metas[:train_data_len], metas[train_data_len:]
    else:
        return metas


class Feeder():
    def __init__(self, coordinator, args, metas=None, name="", multi=4):
        self._coord = coordinator
        self.args = args

        if metas is None:
            metas = args.metadata_dir

        if type(metas) is str:
            self.metadata = load_metadata(metas)
        else:
            self.metadata = metas
        self.n_examples = len(self.metadata)

        with tf.device('/cpu:0'):

            self._placeholders = [
                tf.placeholder(tf.float32, [None, args.n_mel, None]),
                tf.placeholder(
                    tf.float32, [None, args.squeeze_size, args.wav_time_step // args.squeeze_size])
            ]

            queue = tf.FIFOQueue(multi, [tf.float32, tf.float32])
            self._enqueue_op = queue.enqueue(self._placeholders)
            self.mel_inputs, self.wav_inputs = queue.dequeue()
            self.mel_inputs.set_shape(self._placeholders[0].shape)
            self.wav_inputs.set_shape(self._placeholders[1].shape)

    def __main_read(self):
        args = self.args
        wavs_shape = [args.batch_size, args.wav_time_step //
                      args.squeeze_size, args.squeeze_size]

        def randomint():
            return random.randint(0, self.n_examples - 1)

        while not self._coord.should_stop():
            mels, wavs = [], []
            for _ in range(args.batch_size):
                wavp, melp = self.metadata[randomint()]
                mel, wav = padtomaxlen(np.load(melp), np.load(wavp))
                wavs.append(wav)
                mels.append(mel)
            mels = np.transpose(np.array(mels), axes=[0, 2, 1])
            wavs = np.array(wavs)
            wavs = np.reshape(np.array(wavs), wavs_shape)
            wavs = np.transpose(wavs, axes=[0, 2, 1])

            feed_dict = dict(zip(self._placeholders, (mels, wavs)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def start(self, session):
        self._session = session
        thread = threading.Thread(name='background', target=self.__main_read)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

    def dequeue(self):
        mels, wavs = self._session.run([self.mel_inputs, self.wav_inputs])
        return mels, wavs


def mel_iter(meta_path, size=16):
    metadata = load_metadata(meta_path)
    for i in range(0, len(metadata), size):
        yield [np.transpose(np.load(melp)) for wavp, melp in metadata[i:i + size]]
