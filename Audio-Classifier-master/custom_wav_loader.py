import os
import os.path

import librosa
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
AUDIO_EXTENSIONS = ['.wav', '.WAV', ]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def find_classes(dir):
    """

    :param dir:
    :return:
    返回类别，以及类别+每个类别对应的序号
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_index = {classes[i]: i for i in range(len(classes))}
    print("find_classes complete")
    return classes, class_index


def make_dataset(dir, class_index):
    """
    :param dir:当前文件的目录
    :param class_index: 类别和数字对应字典
    :return:
    """
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_index[target])
                    spects.append(item)
    print("make_dataset complete")
    return spects

def logmeldelta_loader(path, window_size=0.02, window_stride=0.01, window_type="hanming", normalize=1, max_len=256):
    """
    在dcase2018 task中截取的音频长度为1.5s 同时在音频长度中随机抽样来判断
    :param path:
    :param window_size:
    :param window_stride:
    :param window_type:
    :param normalize:
    :param max_len:
    :return:返回一个（1，3，64，512）的tensor
    """
    sr = 22050
    try:
        y, sr = librosa.load(path, sr=22050)
    except Exception as e:
        print("waveload error occur:" + str(e) + " Error file is " + path)
        eFile = path
        os.remove(eFile)
    # User set paramters
    if sr is None:
        print("waveload error sr  is None Error file is " + path + "loaded y is" + str(y))
        eFile = path
        os.remove(eFile)
    win_length = 40
    hop_length = 220
    n_fft = int(win_length / 2000 * sr)
    n_mels_bands = 64  # num of mel features
    #对maxlen进行调节
    if sr>22050:
        max_len=512
    elif sr<=22050:
        max_len=256

    # new log mel
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels_bands)

    logmel = librosa.core.power_to_db(melspec)

    delta = librosa.feature.delta(logmel)
    accelerate = librosa.feature.delta(logmel, order=2)

    feats = np.stack((logmel, delta, accelerate))  # (3, 64, xx)
    spect = torch.FloatTensor(feats)
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

def logmel_loader(path, window_size, window_stride, window_type, normalize=1, max_len=256):
    """
    返回一个（1，64，512）的tensor
    :param path:
    :param window_size:
    :param window_stride:
    :param window_type:
    :param normalize:
    :param max_len:
    :return:
    """
    sr = 22050
    try:
        y, sr = librosa.load(path, sr=sr)
    except Exception as e:
        print("waveload error occur:" + str(e) + " Error file is " + path)
        eFile = path
        os.remove(eFile)
    # User set paramters
    if sr is None:
        print("waveload error sr  is None Error file is " + path + "loaded y is" + str(y))
        eFile = path
        os.remove(eFile)
    if sr>22050:
        max_len=512
    elif sr<=22050:
        max_len=256
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    n_mels_bands = 64

    #logmel - band energies

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
        spect = spect[:,offset:(max_len + offset)]
    #spect=spect.T #加转置后在模型的avgpool上会出现bug
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect

def stft_loader(path, window_size, window_stride, window_type, normalize=1, max_len=256):
    sr = 22050
    try:
        y, sr = librosa.load(path, sr=22050)
    except Exception as e:
        print("waveload error occur:" + str(e) + " Error file is " + path)
        eFile = path
        os.remove(eFile)
    # User set paramters
    if sr is None:
        print("waveload error sr  is None Error file is " + path + "loaded y is" + str(y))
        eFile = path
        os.remove(eFile)

    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    n_mels_bands = 64

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_type)
    spect = librosa.amplitude_to_db(D)
    # S = log(S+1)
    # spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        max_offset = spect.shape[1] - max_len
        offset = np.random.randint(max_offset)
        spect = spect[:, :, offset:(max_len + offset)]
    spect=spect.T
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
        class_index (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02, window_stride=.01,
                 window_type='hamming', normalize=True, max_len=256,loader_type="logmeldelta"):

        classes, class_index = find_classes(root)
        spects = make_dataset(root, class_index)
        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_index = class_index
        self.transform = transform
        self.target_transform = target_transform
        if loader_type=="logmeldelta":
            self.loader = logmeldelta_loader
        elif loader_type=="logmel":
            self.loader=logmel_loader
        elif loader_type=="stft":
            self.loader=stft_loader
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
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return spect, target

    def __len__(self):
        return len(self.spects)

    def getClass2Index(self):
        #print(str(self.class_to_idx))
        return self.class_index

class ToTensor(object):
    """
    convert ndarrays in sample to Tensors.
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)

    """
    def __call__(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data


if __name__ == "__main__":
    # from custom_wav_loader import wavLoader
    import torch


    logmeldelta_loader(r"D:\Dataset\UrbanSound\train\gun_shot\_7066.wav",window_size=0.02,window_stride=0.01,window_type="")
    dataset = wavLoader(root=r'D:\Dataset\UrbanSound\codetest')

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=None, num_workers=4, pin_memory=True,
                                              sampler=None)
    print("test_loader:" + str(test_loader.__len__()))

    for k, (input, label) in enumerate(test_loader):
        print(input.size(), len(label))
        print(label)
