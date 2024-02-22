import datetime
import socket
import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
import wandb
import torch
from datasets import load_dataset
from datetime import timedelta, datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, default_data_collator, get_cosine_schedule_with_warmup
from transformers import LlamaForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            
            logits = outputs.logits[:,:-1,:].reshape(-1, outputs.logits.shape[-1])
            labels = batch["labels"][:, 1:].reshape(-1)
            # print(logits.shape) # torch.Size([bs*seq_len, vocab_size])
            # print(labels.shape) # torch.Size([bs*seq_len])

            probs = torch.softmax(torch.tensor(logits), dim = -1) 
            pred = torch.argmax(probs, dim = -1)
            
            gathered_losses = accelerator.gather(loss)
            gathered_pred = accelerator.gather(pred)
            gathered_labels = accelerator.gather(labels)

            acc = accuracy_score(gathered_labels.cpu(), gathered_pred.cpu())
            
            total_loss += gathered_losses.sum().item()
            total_samples += gathered_losses.numel()

    avg_loss = total_loss / total_samples

    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return perplexity, acc


def main(args):

    host = socket.gethostname()
    if 'lovelace' in host:
        output_dir = f"/home/hanshis/workspace/LongContextInfer/archive/ckpts/{args.outputdir}"
        datasetparent = f"/home/hanshis/workspace/Train/data/{args.datadir}/"
        d_files = ["c4_file{}.json".format(i) for i in range(30)]
    else:
        output_dir = f"/fsx-storygen/beidic/hanshi/ckpts/{args.outputdir}"
        datasetparent = f"/fsx-storygen/beidic/hanshi/data/{args.datadir}/"
        d_files = os.listdir(datasetparent)

    os.makedirs(output_dir, exist_ok=True)

    if args.wandb:
        wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout]
    )

    accelerator.init_trackers(
        project_name=args.wandb,
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "gradient_accumulate_every": args.gradient_accumulate_every,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "outputdir": args.outputdir,
            "datadir": args.datadir,
        },
        init_kwargs={"wandb":{"name":args.outputdir}}
    )

    dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")

    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m", legacy=False, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    accelerator.print(f"[{datetime.now()}] Tokenizing dataset...")
    def encode(sample):
        return tokenizer(sample["text"], truncation=True, return_attention_mask = True, max_length=args.datalen, padding="max_length")
    dataset = dataset.map(encode, batched=True, remove_columns=dataset.column_names, num_proc=16)

    accelerator.print(f"[{datetime.now()}] Adding labels to dataset...")
    if "labels" not in dataset.column_names:
        def add_labels(sample):
            sample["labels"] = copy.deepcopy(sample["input_ids"])
            return sample
        dataset = dataset.map(add_labels, desc="Adding labels", num_proc=16)

    # split the dataset
    train_test_split = dataset.train_test_split(test_size=0.001, seed=args.seed)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        shuffle=False,
        batch_size=args.batch_size
    )

    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    raw_max_train_steps = args.epochs * len(train_loader)

    model, optim, train_loader, test_loader = accelerator.prepare(model, optim, train_loader, test_loader)

    max_train_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optim, num_training_steps=raw_max_train_steps, num_warmup_steps=16+int(0.01*raw_max_train_steps))
    scheduler = accelerator.prepare(scheduler)
    accelerator.register_for_checkpointing(scheduler)
    checkpointing_steps = max_train_steps // 4

    total_batch_size = (args.batch_size * accelerator.num_processes * args.gradient_accumulate_every)
    accelerator.print("============================== Configurations ==============================")
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")
    accelerator.print("Trainset samples: ", len(train_dataset), "Testset samples: ", len(test_dataset))
    accelerator.print(f"Max train steps: {max_train_steps}, Epochs: {max_train_steps / len(train_loader)}")
    accelerator.print(f"Real batch size: {total_batch_size}, bathes per epoch: {len(train_loader)}, gradient accumulate every: {args.gradient_accumulate_every}")
    accelerator.print(f"Init learning rate: {args.learning_rate}, Warmup steps: {16+int(0.01*max_train_steps)}")
    accelerator.print(f"Checkpointing steps: {checkpointing_steps}")
    accelerator.print(f"Input shape len (double check): {len(train_dataset[0]['input_ids'])}")
    accelerator.print("============================================================================")

    ###### begin training eval ######
    ppl, acc = evaluate(model, test_loader, accelerator)
    accelerator.print(f"[{datetime.now()}] Initial perplexity before Training: {ppl}, Initial accuracy before Training: {acc}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.epochs):

        ##################### Training #####################
        model.train()
        for _, batch in enumerate(train_loader):
            with accelerator.accumulate(model):

                loss = model(**batch).loss
                accelerator.backward(loss)
                optim.step()
                scheduler.step()
                optim.zero_grad()

                if accelerator.sync_gradients:
                    
                    ppl, acc = evaluate(model, test_loader, accelerator)
                    
                    loss_log = {
                        "loss": loss.item(),
                        "lr": optim.param_groups[0]["lr"],
                        "ppl": ppl,
                        "acc": acc,
                    }
                    accelerator.log(loss_log, step=completed_steps)
                    
                    if completed_steps % 10 == 0:
                        progress_bar.update(10)
                        progress_bar.set_postfix(loss_log)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int) and completed_steps > 0:
                    if completed_steps % checkpointing_steps == 0:
                        accelerator.save_state(os.path.join(output_dir, f"step_{completed_steps}"))
            
            if completed_steps >= max_train_steps:
                break

        accelerator.print(f"[{datetime.now()}] Epoch {epoch+1} / {args.epochs} finished, ppl: {ppl}, acc: {acc}")

    accelerator.print(f"[{datetime.now()}] Training Finished, final ppl: {ppl}")
    accelerator.end_training()

    accelerator.print(f"[{datetime.now()}] Saving model to {args.outputdir}")
    accelerator.wait_for_everyone()

    state_dict = accelerator.get_state_dict(model, unwrap=False)
    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )

    accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--wandb", type=str, default=None)
    args.add_argument("--outputdir", type=str, default=None)
    args.add_argument("--datadir", type=str, default=None)
    args.add_argument("--datalen", type=int, default=256)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--learning-rate", type=float, default=3e-4)
    args.add_argument("--model", type=str, default="JackFram/llama-68m")
    main(args.parse_args())
