from transformers.integrations import WandbCallback
import pandas as pd
from torch.utils.data import Subset
from transformers import Seq2SeqTrainer
import evaluate
import wandb
import torch
from torch.utils.data import DataLoader
import wandb
from PIL import Image
from torchvision import transforms
from utils.data_utils import DataCollatorBrainSpeechSeq2SeqWithPadding
from transformers import PreTrainedModel, PretrainedConfig, WhisperProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from allennlp.training.callbacks import TrainerCallback, WandBCallback

def decode_predictions(tokenizer, predictions, metric_file):
    labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    logits = predictions.predictions[0].argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits, skip_special_tokens=True)
    print("labels:", labels, "predictions:", prediction_text)
    metric = evaluate.load(f'metrics/{metric_file}.py')
    metric.add_batch(predictions=prediction_text, references=labels)
    score = metric.compute()
    return {"labels": labels, "predictions": prediction_text}, score

def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 4))
    sns.heatmap(spectrogram.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()

def concatenate_images(pred_mel, gt_mel):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    sns.heatmap(pred_mel.cpu().numpy(), cmap='viridis', ax=axs[0])
    axs[0].set_title('Predicted Mel-Spectrogram')

    sns.heatmap(gt_mel.cpu().numpy(), cmap='viridis', ax=axs[1])
    axs[1].set_title('Ground Truth Mel-Spectrogram')

    # Remove axis for a cleaner look
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

    # Convert to PIL image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return Image.fromarray(img)

class WandbPredictionProgressCallback(WandbCallback):

    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, train_dataset,
                 num_samples=100, freq=5):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.processor = WhisperProcessor.from_pretrained('openai/whisper-base',
                                                    language='en',
                                                    task='transcribe',
                                                    local_files_only=False)
        self.device=torch.device(f"cuda:0")
        self.sample_dataset = Subset(val_dataset, range(num_samples))
        self.valid_dataloader = DataLoader(self.sample_dataset, batch_size=1, collate_fn=DataCollatorBrainSpeechSeq2SeqWithPadding(processor=self.processor))
        self.sample_dataset_train = Subset(train_dataset, range(num_samples))
        self.train_dataloader = DataLoader(self.sample_dataset_train, batch_size=1, collate_fn=DataCollatorBrainSpeechSeq2SeqWithPadding(processor=self.processor))

        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        print("on_evaluate start")
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        # print(self.sample_dataset)
        if state.global_step % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            self.trainer.model.eval()
            example_images = []
            for batch in self.valid_dataloader:
                with torch.no_grad():
                    '''validation set'''
                    output = self.trainer.model(input_features = batch['input_features'].to(self.device), 
                                                useful_length=batch['useful_length'],
                                                attention_mask=batch['attention_mask'].to(self.device),
                                                mel_spec=batch['mel_spec'].to(self.device),
                                                labels=batch['labels'].to(self.device),
                                                subject_index=batch['subject_index'].to(self.device)
                                                )
                    # Transform tensors to PIL images
                    pred_mel = output.p_mel.squeeze(0)  
                    gt_mel = output.mel.squeeze(0)  
                    combined_img = concatenate_images(pred_mel, gt_mel)
                    example_images.append(wandb.Image(combined_img, caption="Pred and GT [evaluation set]"))

            for batch in self.train_dataloader:
                with torch.no_grad():
                    '''train set'''
                    output_train = self.trainer.model(input_features = batch['input_features'].to(self.device), 
                                                useful_length=batch['useful_length'],
                                                attention_mask=batch['attention_mask'].to(self.device),
                                                mel_spec=batch['mel_spec'].to(self.device),
                                                labels=batch['labels'].to(self.device),
                                                subject_index=batch['subject_index'].to(self.device)
                                                )
                    # Transform tensors to PIL images
                    pred_mel_train = output_train.p_mel.squeeze(0)  
                    gt_mel_train = output_train.mel.squeeze(0)  
                    combined_img_train = concatenate_images(pred_mel_train, gt_mel_train)
                    example_images.append(wandb.Image(combined_img_train, caption="Pred and GT [train set]"))

            # Log the images to wandb
            self._wandb.log({"MEL spectrogram": example_images})
            
            # decode predictions and labels
            predictions, score = decode_predictions(self.tokenizer, predictions, 'bleu')
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.global_step
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"bleu": score})
            self._wandb.log({"sample_predictions": records_table})



    def on_substep_end(self, args, state, control, logs=None, **kwargs):
        super().on_substep_end(args, state, control, logs=logs, **kwargs)
        print('on_substep_end')
        '''
        if state.global_step:
            print(logs)
            if logs is not None:
                if "clip" in logs:
                    self._wandb.log({"clip": logs.get("clip")})
                if "mmd" in logs:
                    self._wandb.log({"mmd": logs.get("mmd")})
                if "ce" in logs:
                    self._wandb.log({"ce": logs.get("ce")})
        '''
                


class WandbPredictionProgressCallback_test(WandbCallback):

    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, train_dataset,
                 num_samples=100, freq=5):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.processor = WhisperProcessor.from_pretrained('openai/whisper-base',
                                                    language='en',
                                                    task='transcribe',
                                                    local_files_only=False)
        self.device=torch.device(f"cuda:0")
        self.sample_dataset = Subset(val_dataset, range(num_samples))
        self.valid_dataloader = DataLoader(self.sample_dataset, batch_size=1, collate_fn=DataCollatorBrainSpeechSeq2SeqWithPadding(processor=self.processor))
        self.sample_dataset_train = Subset(train_dataset, range(num_samples))
        self.train_dataloader = DataLoader(self.sample_dataset_train, batch_size=1, collate_fn=DataCollatorBrainSpeechSeq2SeqWithPadding(processor=self.processor))

        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        print("on_evaluate start")
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        # print(self.sample_dataset)
        if state.global_step % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            self.trainer.model.eval()
            example_images = []
            for batch in self.valid_dataloader:
                with torch.no_grad():
                    '''validation set'''
                    output = self.trainer.model(input_features = batch['input_features'].to(self.device), 
                                                useful_length=batch['useful_length'],
                                                attention_mask=batch['attention_mask'].to(self.device),
                                                mel_spec=batch['mel_spec'].to(self.device),
                                                labels=batch['labels'].to(self.device),
                                                subject_index=batch['subject_index'].to(self.device)
                                                )
                    # Transform tensors to PIL images
                    pred_mel = output.p_mel.squeeze(0)  
                    gt_mel = output.mel.squeeze(0)  
                    combined_img = concatenate_images(pred_mel, gt_mel)
                    example_images.append(wandb.Image(combined_img, caption="Pred and GT [evaluation set]"))

            for batch in self.train_dataloader:
                with torch.no_grad():
                    '''train set'''
                    output_train = self.trainer.model(input_features = batch['input_features'].to(self.device), 
                                                useful_length=batch['useful_length'],
                                                attention_mask=batch['attention_mask'].to(self.device),
                                                mel_spec=batch['mel_spec'].to(self.device),
                                                labels=batch['labels'].to(self.device),
                                                subject_index=batch['subject_index'].to(self.device)
                                                )
                    # Transform tensors to PIL images
                    pred_mel_train = output_train.p_mel.squeeze(0)  
                    gt_mel_train = output_train.mel.squeeze(0)  
                    combined_img_train = concatenate_images(pred_mel_train, gt_mel_train)
                    example_images.append(wandb.Image(combined_img_train, caption="Pred and GT [train set]"))

            # Log the images to wandb
            self._wandb.log({"MEL spectrogram": example_images})
            
            # decode predictions and labels
            predictions, score = decode_predictions(self.tokenizer, predictions, 'bleu')
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.global_step
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"bleu": score})
            self._wandb.log({"sample_predictions": records_table})



    def on_substep_end(self, args, state, control, logs=None, **kwargs):
        super().on_substep_end(args, state, control, logs=logs, **kwargs)
        print('on_substep_end')
        '''
        if state.global_step:
            print(logs)
            if logs is not None:
                if "clip" in logs:
                    self._wandb.log({"clip": logs.get("clip")})
                if "mmd" in logs:
                    self._wandb.log({"mmd": logs.get("mmd")})
                if "ce" in logs:
                    self._wandb.log({"ce": logs.get("ce")})
        '''
                

