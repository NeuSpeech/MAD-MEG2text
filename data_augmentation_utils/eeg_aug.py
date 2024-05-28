import numpy as np
import random
from utils.augment_eeg import RandomShapeMasker
from utils.utils import add_gaussian_noise


# 首先写一些底层函数用于调用，
# 然后写一个类来方便调用，这个类可以更方便的使用config

# 目标实现的功能有：改变MEG信号的长度都相同的处境
# 可以使用以下方法实现该功能
# 1，对MEG信号进行速度改变，即改变采样率
# 2，将MEG信号进行多段的速度改变，即不同段的速度不同
# 3，将MEG信号进行多段的切割，插入空白
# 4，将MEG信号进行多段的切割，插入等功率的白噪声
# 5，将MEG信号进行平滑变速，即有一个变速曲线，根据这个曲线来加速信号
# 现在你先写出我们要实现哪些最基础的函数，
# 即组合起来就可以实现上述5个方法的函数。
# MEG信号的尺寸是[channel,length]
# 采样率是sampling_rate
#

# 假设输入的MEG信号是一个NumPy数组，形状为[channel, length]
# 1. 改变采样率（速度改变）
def change_sampling_rate(signal, speed_factor):
    # 计算目标长度
    # print(f'signal shape:{signal.shape}')
    target_length = int(signal.shape[1] / speed_factor)
    # 使用numpy的resample函数进行重采样
    resampled_signal = np.zeros((signal.shape[0], target_length))

    for ch in range(signal.shape[0]):
        resampled_signal[ch] = np.interp(np.linspace(0, signal.shape[1] - 1, target_length), np.arange(signal.shape[1]),
                                         signal[ch])
    return resampled_signal


# 2. 多段速度改变
def multi_segment_speed_change(signal, segment_endpoints, segment_speeds):
    current_index = 0
    processed_signal = []
    for i, endpoint in enumerate(segment_endpoints):
        # 获取当前段的信号
        segment_signal = signal[:, current_index:endpoint]
        # 获取当前段的速度因子
        speed_factor = segment_speeds[i]
        # 改变采样率
        changed_segment = change_sampling_rate(segment_signal, speed_factor)
        # 拼接处理后的信号
        processed_signal.append(changed_segment)
        current_index = endpoint
    processed_signal = np.concatenate(processed_signal, axis=1)
    return processed_signal


def cut_signal(signal, cut_points):
    segments = []
    start = 0
    for point in cut_points:
        segments.append(signal[:, start:point])
        start = point
    segments.append(signal[:, start:])
    return segments


def insert_content(signal, insert_contents, insert_points):
    assert len(insert_contents) == len(insert_points)
    segments = cut_signal(signal, insert_points)
    assert len(segments) > 1
    full_segs = []
    # 使用np.concatenate来合并信号片段和插入内容
    for i, content in enumerate(insert_contents):
        full_segs.extend([segments[i], content])
    full_segs.append(segments[-1])
    return np.concatenate(full_segs, axis=1)


def calculate_power(signal):
    # 计算每个通道的功率
    power = np.mean(signal ** 2, axis=1)
    return power


def generate_white_noise(power, length, channel_count):
    # 假设power是每个通道的功率，length是噪声的长度
    std_dev = np.sqrt(power)[:, None]  # 功率等于标准差的平方
    noise = np.random.normal(0, 1, size=(channel_count, length))
    noise = std_dev * noise
    return noise


# 3. 插入空白
def insert_blanks(signal, insert_lengths, insert_points):
    blank_segments = []
    channel_count = signal.shape[0]
    for length in insert_lengths:
        blank_segment = np.zeros((channel_count, length))
        blank_segments.append(blank_segment)
    return insert_content(signal, blank_segments, insert_points)


# 4. 插入等功率的白噪声
def insert_white_noise(signal, insert_lengths, insert_points):
    noise_segments = []
    channel_count = signal.shape[0]
    for length in insert_lengths:
        # 计算每个通道的功率，上下1dB的抖动
        power = calculate_power(signal) * (1 + np.random.uniform(-0.1, 0.1, channel_count))
        noise_segment = generate_white_noise(power, length, channel_count)
        noise_segments.append(noise_segment)
    # 将噪声片段与信号片段拼接
    return insert_content(signal, noise_segments, insert_points)


# 5. 平滑变速
def smooth_speed_change(signal, speed_curve):
    # 计算新的采样点
    new_sampling_points = np.cumsum(speed_curve)
    new_sampling_points = new_sampling_points * signal.shape[1] / np.max(new_sampling_points)
    # 使用numpy的interp函数根据新的采样点对信号进行插值
    interpolated_signal = [np.interp(new_sampling_points, np.arange(signal.shape[1]), signal[ch])
                           for ch in range(signal.shape[0])]
    interpolated_signal = np.stack(interpolated_signal, axis=0)

    return interpolated_signal


class EEGAug:
    def __init__(self,
                 max_signal_duration=30, sampling_rate=200,
                 min_speed=0.8, max_speed=1.2,
                 min_segments=2, max_segments=10,
                 min_inserts=0, max_inserts=10,
                 # mask config
                 unit=(1, 40), mask_prob=0.5,
                 # add noise
                 min_snr_db=10, max_snr_db=20,
                 # volume
                 min_gain=0, max_gain=20,

                 ):
        self.max_signal_length = int(max_signal_duration * sampling_rate)
        self.sampling_rate = sampling_rate
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_segments = max_segments
        self.min_segments = min_segments
        self.max_inserts = max_inserts
        self.min_inserts = min_inserts
        self.unit = unit
        self.mask_prob = mask_prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.funcs = ['speed_change', 'multi_speed_change', 'smooth_speech_change',
                      'insert_blank', 'insert_white_noise',
                      'block_mask', 'time_mask', 'channel_mask',
                      'add_white_noise',
                      'volume_change',
                      ]

    def speed_change(self, signal):
        random_speed = random.uniform(self.min_speed, self.max_speed)
        return change_sampling_rate(signal, random_speed)

    def multi_speed_change(self, signal):
        # 计算最大可加速倍率
        max_acceleration_factor = self.max_signal_length / signal.shape[1]
        # 确定上限倍率
        max_speed_factor = min(self.max_speed, max_acceleration_factor)
        num_segments = random.randint(self.min_segments, self.max_segments)
        segment_endpoints = np.sort(np.random.randint(0, signal.shape[1], size=num_segments))
        segment_length = np.diff(np.array([0, *segment_endpoints]))
        threshold = 20
        segment_endpoints_new = [segment_endpoints[i] for i, seg in enumerate(segment_length) if seg > threshold]

        segment_speeds = np.random.uniform(self.min_speed, max_speed_factor, size=len(segment_endpoints_new))
        if len(segment_endpoints_new) == 0:
            return signal
        else:
            return multi_segment_speed_change(signal, segment_endpoints_new, segment_speeds)

    def _calc_args_insert(self, signal):
        num_inserts = random.randint(self.min_inserts, self.max_inserts)
        max_insert_length = np.random.randint(low=num_inserts + 1, high=self.max_signal_length - signal.shape[1],
                                              size=None)
        insert_lengths_cum = np.sort(np.random.randint(1, max_insert_length, size=num_inserts))
        insert_lengths = np.diff(insert_lengths_cum)
        insert_lengths = np.insert(insert_lengths, 0, insert_lengths_cum[0])
        insert_points = np.sort(np.random.randint(0, self.max_signal_length, size=num_inserts))
        return insert_lengths, insert_points

    def insert_blank(self, signal):
        insert_lengths, insert_points = self._calc_args_insert(signal)
        return insert_blanks(signal, insert_lengths, insert_points)

    def insert_white_noise(self, signal):
        insert_lengths, insert_points = self._calc_args_insert(signal)
        return insert_white_noise(signal, insert_lengths, insert_points)

    def smooth_speech_change(self, signal):
        min_speed = signal.shape[1] / self.max_signal_length
        min_speed = max(min_speed, self.min_speed)
        speed_curve = np.random.uniform(min_speed, self.max_speed, signal.shape[1])
        return smooth_speed_change(signal, speed_curve)

    def mask(self, signal, mask_type):
        augmentor = RandomShapeMasker(
            unit=self.unit, mask_prob=self.mask_prob, mask_type=mask_type
        )
        mask = augmentor(signal.shape)
        return signal * mask

    def add_white_noise(self, signal):
        signal = add_gaussian_noise(signal, snr_range=(self.min_snr_db, self.max_snr_db))
        return signal

    def volume_change(self, signal):
        gain = np.random.rand() * (self.max_gain - self.min_gain) + self.min_gain
        signal = signal * 10. ** (gain / 20.)
        return signal

    def __call__(self, signal, func):
        assert func in self.funcs
        if func == "speed_change":
            return self.speed_change(signal)
        elif func == "multi_speed_change":
            return self.multi_speed_change(signal)
        elif func == "insert_blank":
            return self.insert_blank(signal)
        elif func == "insert_white_noise":
            return self.insert_white_noise(signal)
        elif func == "smooth_speech_change":
            return self.smooth_speech_change(signal)
        elif func == "block_mask":
            return self.mask(signal, mask_type=1)
        elif func == "time_mask":
            return self.mask(signal, mask_type=2)
        elif func == "channel_mask":
            return self.mask(signal, mask_type=3)
        elif func == "add_white_noise":
            return self.add_white_noise(signal)
        elif func == "volume_change":
            return self.volume_change(signal)
        else:
            raise ValueError(f"Unknown function name: {func}")
