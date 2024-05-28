import os
import shutil

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

class SavePeftModelCallback1(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # First, check whether it is time to save
        control.should_save=False
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        
        if state.global_step % state.save_steps == 0:
            log_history = []
            for i, data in enumerate(state.log_history):
                if 'eval_loss' in data.keys():
                    log_history.append(data['eval_loss'])
            if len(log_history)>0:
                if log_history[-1]==min(log_history):
                    control.should_save=True
                    print('should_save')
                    if not os.path.exists(checkpoint_folder):
                        os.makedirs(checkpoint_folder)
                    torch.save(kwargs["model"], checkpoint_folder + "/full_model.pth")
                    #kwargs["model"].base_model.save_pretrained(checkpoint_folder)
                    #model.base_model.save_pretrained(xxx)

                # if os.path.exists(state.best_model_checkpoint):
                #     shutil.rmtree(state.best_model_checkpoint)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # No saving allowed after epoch ends
        control.should_save=False

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save=False

    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
            # Copy the Lora model, mainly to be compatible with old versions of peft
        # args.should_save=False
        
        # checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        # if os.path.exists(checkpoint_folder):
        #     shutil.rmtree(checkpoint_folder)
        # if len(state.log_history)>0:
        #     now_metric=state.log_history[-1]['eval_loss']
        # else:
        #     now_metric=None
        # if now_metric==state.best_metric:
        #     # this is the best
        #     kwargs["model"].save_pretrained(checkpoint_folder)
        #     write_jsonlines(f'{checkpoint_folder}/history.txt',state.log_history)
        # print('checkpoint_folder', checkpoint_folder)
        # kwargs["model"].save_pretrained(checkpoint_folder)
        return control
    
# Callback function when saving the model
class SavePeftModelCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # First, check whether it is time to save
        control.should_save=False
        if state.global_step % state.save_steps == 0:
            log_history = []
            for i, data in enumerate(state.log_history):
                if 'eval_loss' in data.keys():
                    log_history.append(data['eval_loss'])
            if len(log_history)>0:  # Because after training for a long time, best_metric will become null, so you have to do it yourself.
                if log_history[-1]==min(log_history):
                    control.should_save=True

                # if os.path.exists(state.best_model_checkpoint):
                #     shutil.rmtree(state.best_model_checkpoint)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # No saving allowed after epoch ends
        control.should_save=False

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save=False

    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
            # Copy the Lora model, mainly to be compatible with old versions of peft
        # args.should_save=False
        # checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        # if os.path.exists(checkpoint_folder):
        #     shutil.rmtree(checkpoint_folder)
        # if len(state.log_history)>0:
        #     now_metric=state.log_history[-1]['eval_loss']
        # else:
        #     now_metric=None
        # if now_metric==state.best_metric:
        #     kwargs["model"].save_pretrained(checkpoint_folder)
        #     write_jsonlines(f'{checkpoint_folder}/history.txt',state.log_history)
        return control

class SaveFullModelCallback(TrainerCallback):
    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
        if args.local_rank == 0 or args.local_rank == -1:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_folder)
            # Save the best model
            best_checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")
            # Because only the latest 5 checkpoints are saved, make sure they are not previous checkpoints.
            if os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(f"The checkpoints that work best are: {state.best_model_checkpoint}, the evaluation result is: {state.best_metric}")
        return control


