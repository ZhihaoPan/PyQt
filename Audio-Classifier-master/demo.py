import os, time
import os.path

import librosa
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

AUDIO_EXTENSIONS = ['.wav', '.WAV', ]
allLabel = {'baby_cry': 0, 'car_engine': 1, 'crowd': 2, 'dog_bark': 3, 'gun_shot': 4, 'multispeaker': 5, 'scream': 6,
             'siren': 7, 'speaking': 8, 'stLaughter': 9}


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset(dir):
    start = time.clock()
    y, sr = librosa.load(dir, sr=22050)
    # print(y.shape)
    if sr == 22050:
        trucateLen = 55000
    else:
        trucateLen = 125100
    y_len = len(y)
    spects = []
    temp = 0
    if y_len > trucateLen:
        while y_len > trucateLen:
            item = (dir, temp * trucateLen)
            temp += 1
            y_len = y_len - trucateLen
            spects.append(item)
    else:
        item = (dir, 0)
        spects.append(item)

    print("Cut demo audio complete")
    end = time.clock()
    print("reading file time:{:.3f}s".format(end - start))
    return spects, y, sr, trucateLen


def logmeldelta_loader(y, sr, begin, trucateLen, window_size, window_stride, window_type, normalize, max_len=256):
    win_length = 40
    hop_length = 220
    n_fft = int(win_length / 2000 * sr)
    n_mels_bands = 64  # num of mel features

    y = y[begin:begin + trucateLen]
    # new log mel
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels_bands)

    logmel = librosa.core.power_to_db(melspec)

    delta = librosa.feature.delta(logmel)
    accelerate = librosa.feature.delta(logmel, order=2)

    feats = np.stack((logmel, delta, accelerate))  # (3, 64, xx)
    spect = torch.FloatTensor(feats)
    # print(spect.shape)
    # 设为定长
    if spect.shape[2] > max_len:
        max_offset = spect.shape[2] - max_len
        offset = np.random.randint(max_offset)
        spect = spect[:, :, offset:(max_len + offset)]
    else:
        if max_len > spect.shape[2]:
            max_offset = max_len - spect.shape[2]
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        spect = np.pad(spect, ((0, 0), (0, 0), (offset, max_len - spect.shape[2] - offset)), "constant")
    spect = torch.FloatTensor(spect)

    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect


def logmel_loader(y, sr, begin, trucateLen, window_size, window_stride, window_type, normalize, max_len=512):
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    n_mels_bands = 64
    y = y[begin:begin + trucateLen]
    # logmel - band energies

    spect, _n_fft = librosa.core.spectrum._spectrogram(y=y, n_fft=n_fft, hop_length=hop_length, power=1)

    # mel filter
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels_bands)
    spect = np.log1p(np.dot(mel_basis, spect), dtype=np.float32)

    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        max_offset = spect.shape[1] - max_len
        offset = np.random.randint(max_offset)
        spect = spect[:, offset:(max_len + offset)]
    # spect=spect.T
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect


def stft_loader(y, sr, begin, trucateLen, window_size, window_stride, window_type, normalize, max_len=512):
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    y = y[begin:begin + trucateLen]
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_type)
    spect = librosa.amplitude_to_db(D)
    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = spect.T
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect


class wavLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use !!! if the audio is all 4 second then can be set to 401
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, dir, allLabels, transform=None, target_transform=None, window_size=.02, window_stride=.01,
                 window_type='hamming', normalize=True, max_len=256, loader_type="logmeldelta"):
        spects, self.data, self.sr, self.trucatelen = make_dataset(dir)
        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        # self.classes = classes
        self.class_to_idx = allLabels
        self.transform = transform
        self.target_transform = target_transform
        if loader_type == "logmeldelta":
            self.loader = logmeldelta_loader
        elif loader_type == "logmel":
            self.loader = logmel_loader
        elif loader_type == "stft":
            self.loader = stft_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, begin = self.spects[index]
        spect = self.loader(self.data, self.sr, begin, self.trucatelen, self.window_size, self.window_stride,
                            self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return spect, begin

    def __len__(self):
        return len(self.spects)

    def getClass2Index(self):
        return self.class_to_idx


if __name__ == "__main__":
    import torch

    make_dataset(
        r"D:\Dataset\UrbanSound\train\gun_shot\_7066.wav")  # dataset = wavLoader(root=r'D:\Dataset\UrbanSound\codetest',dir=r"D:\Dataset\UrbanSound\complete_data\children_playing\69962.wav")  #  # test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=None, num_workers=4, pin_memory=True,  #                                           sampler=None)  #
