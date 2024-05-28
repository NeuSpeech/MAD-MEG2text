import random
from data_augmentation_utils.eeg_aug import EEGAug
from data_augmentation_utils.eeg_text_aug import EEGTextAug
from data_augmentation_utils.text_augmentation import TextAug


class FullAug:
    def __init__(self, configs=None,augmentations=()):
        if configs is not None and 'augmentations' in configs and configs['augmentations'] is not None:
            pass

    def __init__(self, configs=dict(), augmentations=()):
        if 'augmentations' in configs and configs['augmentations'] is not None:
            self.EEG_aug = EEGAug(**configs['common'],**configs['eeg'])
            self.text_aug = TextAug(**configs['text'])
            self.EEG_text_aug = EEGTextAug(**configs['common'],**configs['eeg_text'])
            self.augmentations = configs['augmentations']
        else:
            self.EEG_aug = EEGAug()
            self.text_aug = TextAug()
            self.EEG_text_aug = EEGTextAug()
            self.augmentations = augmentations
        self.allowed_types = self.EEG_aug.funcs + self.text_aug.funcs + self.EEG_text_aug.funcs
        # example for augmentations: [['type',prob],]

    def augment_data(self, unit):
        for aug_type, prob in self.augmentations:
            if random.random() < prob:
                if aug_type in self.EEG_aug.funcs:
                    unit['eeg_raw'] = self.EEG_aug(unit['eeg_raw'],
                                                   func=aug_type)
                    # it is hard to rectify time, so it is not implemented.
                elif aug_type in self.text_aug.funcs:
                    unit['sentence'] = self.text_aug(unit['sentence'],
                                                     func=aug_type)
                elif aug_type in self.EEG_text_aug.funcs:
                    unit = self.EEG_text_aug(unit,aug_type)
        return unit

    def __call__(self, unit):
        return self.augment_data(unit)

