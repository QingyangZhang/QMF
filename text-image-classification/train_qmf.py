#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from src.data.helpers import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *

def get_args(parser: argparse.ArgumentParser) -> None:
    """Defines and parses command line arguments"""
    parser.add_argument("--batch_sz", type=int, default=32, help="Batch size")
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased", help="Pre-trained BERT model path")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Dataset directory path")
    parser.add_argument("--drop_img_percent", type=float, default=0.0, help="Percentage of images to drop")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--embed_sz", type=int, default=300, help="Embedding size")
    parser.add_argument("--freeze_img", type=int, default=3, help="Epochs to freeze image encoder")
    parser.add_argument("--freeze_txt", type=int, default=5, help="Epochs to freeze text encoder")
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt", help="GloVe embeddings path")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24, help="Gradient accumulation steps")
    parser.add_argument("--hidden", nargs="*", type=int, default=[], help="Hidden layer sizes")
    parser.add_argument("--hidden_sz", type=int, default=768, help="Main hidden size")
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"], help="Image embedding pooling method")
    parser.add_argument("--img_hidden_sz", type=int, default=2048, help="Image encoder hidden size")
    parser.add_argument("--include_bn", type=int, default=True, help="Include batch normalization")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate reduction factor")
    parser.add_argument("--lr_patience", type=int, default=2, help="Patience for learning rate scheduler")
    parser.add_argument("--max_epochs", type=int, default=0, help="Maximum training epochs")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--model", type=str, default="latefusion", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt", "latefusion"], help="Model architecture")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--name", type=str, default="trial-03", help="Experiment name")
    parser.add_argument("--num_image_embeds", type=int, default=3, help="Number of image embeddings")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--savedir", type=str, default="./checkpoint", help="Directory to save models")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--task", type=str, default="MVSA_Single", choices=["food101", "MVSA_Single"], help="Task name")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification"], help="Task type")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup proportion for BERT training")
    parser.add_argument("--weight_classes", type=int, default=1, help="Apply class weighting")
    parser.add_argument("--df", type=bool, default=True, help="Use dynamic fusion (if applicable)")
    parser.add_argument("--noise_level", type=float, default=0.5, help="Noise level for testing")
    parser.add_argument("--noise_type", type=str, default='Gaussian', help="Noise type for testing")


def get_criterion(args: argparse.Namespace) -> nn.Module:
    """Create loss function based on task type"""
    if args.task_type == "multilabel":
        if args.weight_classes:
            # Calculate inverse frequency weights for imbalanced classes
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        # Cross-entropy for classification tasks
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    """Create optimizer based on model architecture"""
    if args.model in ["bert", "concatbert", "mmbt"]:
        # Special handling for BERT-based models
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        # Standard Adam for other models
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        patience=args.lr_patience, 
        verbose=True, 
        factor=args.lr_factor
    )


def model_eval(
    epoch: int, 
    dataloader: torch.utils.data.DataLoader, 
    model: nn.Module, 
    args: argparse.Namespace, 
    criterion: nn.Module, 
    store_preds: bool = False
) -> dict:
    """Evaluate model on given dataset"""
    model.eval()
    losses, preds, targets = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating epoch {epoch}"):
            loss, output, target = model_forward(epoch, model, args, criterion, batch, mode='eval')
            losses.append(loss.item())

            if args.task_type == "multilabel":
                # Apply sigmoid and threshold at 0.5 for multi-label classification
                pred = (torch.sigmoid(output).cpu().detach().numpy() > 0.5)
            else:
                # For classification, take argmax after softmax
                pred = torch.nn.functional.softmax(output, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            target = target.cpu().detach().numpy()
            targets.append(target)

    # Calculate metrics
    metrics = {"loss": np.mean(losses)}
    
    if args.task_type == "multilabel":
        targets = np.vstack(targets)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(targets, preds, average="macro")
        metrics["micro_f1"] = f1_score(targets, preds, average="micro")
    else:
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        metrics["acc"] = accuracy_score(targets, preds)

    if store_preds:
        store_preds_to_disk(targets, preds, args)

    return metrics


def rank_loss(
    confidence: torch.Tensor, 
    indices: torch.Tensor, 
    history: 'History'
) -> torch.Tensor:
    """Calculate ranking loss for confidence calibration"""
    # Create paired inputs
    confidence1 = confidence
    confidence2 = torch.roll(confidence, -1)
    indices2 = torch.roll(indices, -1)

    # Get target rankings and margins from history
    rank_target, rank_margin = history.get_target_margin(indices, indices2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    confidence2 = confidence2 + (rank_margin / rank_target_nonzero).reshape((-1, 1))

    # Compute margin ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(
        confidence1,
        confidence2,
        -rank_target.reshape(-1, 1)
    )

    return ranking_loss


def model_forward(
    epoch: int,
    model: nn.Module,
    args: argparse.Namespace,
    criterion: nn.Module,
    batch: tuple,
    txt_history: 'History' = None,
    img_history: 'History' = None,
    mode: str = 'train'
) -> tuple:
    """Perform forward pass and compute losses"""
    text, segment, mask, image, target, indices = batch
    freeze_img = epoch < args.freeze_img
    freeze_txt = epoch < args.freeze_txt

    # Move tensors to GPU
    device = next(model.parameters()).device
    text = text.to(device)
    mask = mask.to(device)
    segment = segment.to(device)
    image = image.to(device)
    target = target.to(device)
    
    # Forward pass for different model architectures
    if args.model == "bow":
        output = model(text)
    elif args.model == "img":
        output = model(image)
    elif args.model == "concatbow":
        output = model(text, image)
    elif args.model == "bert":
        output = model(text, mask, segment)
    elif args.model == "concatbert":
        output = model(text, mask, segment, image)
    elif args.model == "latefusion":
        # Late fusion returns multiple outputs
        text_img_logits, text_logits, image_logits, text_conf, image_conf = model(
            text, mask, segment, image
        )
    else:  # args.model == "mmbt"
        # Handle parameter freezing
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt
        output = model(text, mask, segment, image)

    # Task-specific loss calculations
    if args.model == "latefusion":
        # Calculate modality-specific losses
        text_clf_loss = criterion(text_logits, target)
        image_clf_loss = criterion(image_logits, target)
        joint_clf_loss = criterion(text_img_logits, target)
        
        # Calculate total classification loss
        clf_loss = text_clf_loss + image_clf_loss + joint_clf_loss
        
        # For training mode, calculate losses per sample and update history
        if mode == 'train':
            text_loss_per_sample = nn.CrossEntropyLoss(reduction='none')(text_logits, target).detach()
            image_loss_per_sample = nn.CrossEntropyLoss(reduction='none')(image_logits, target).detach()
            
            # Update history modules with current sample metrics
            txt_history.correctness_update(indices, text_loss_per_sample, text_conf.squeeze())
            img_history.correctness_update(indices, image_loss_per_sample, image_conf.squeeze())
            
            # Compute ranking losses for confidence calibration
            text_rank_loss = rank_loss(text_conf, indices, txt_history)
            image_rank_loss = rank_loss(image_conf, indices, img_history)
            
            # Combine classification and ranking losses
            crl_loss = text_rank_loss + image_rank_loss
            total_loss = torch.mean(clf_loss + crl_loss)
            
            return total_loss, text_img_logits, target
        else:
            return torch.mean(clf_loss), text_img_logits, target
    else:
        # Standard loss computation for other models
        loss = criterion(output, target)
        return loss, output, target


def train(args: argparse.Namespace) -> None:
    """Main training loop"""
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    # Data loading
    train_loader, val_loader, test_loaders = get_data_loaders(args)
    
    # Model initialization
    model = get_model(args)
    model.cuda()

    # Training components
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # Logging and checkpointing setup
    logger = create_logger(os.path.join(args.savedir, "logfile.log"), args)
    logger.info(f"Model architecture:\n{model}")
    torch.save(args, os.path.join(args.savedir, "args.pt"))

    # Resume training if checkpoint exists
    start_epoch, global_step, no_improve_count, best_metric = 0, 0, 0, -np.inf
    checkpoint_path = os.path.join(args.savedir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        no_improve_count = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Starting training process...")
    
    # Initialize history trackers (requires History implementation)
    txt_history = History(len(train_loader.dataset))
    img_history = History(len(train_loader.dataset))

    for epoch in range(start_epoch, args.max_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            loss, _, _ = model_forward(
                epoch, 
                model, 
                args, 
                criterion, 
                batch,
                txt_history,
                img_history,
                mode='train'
            )
            
            # Normalize loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_losses.append(loss.item())
            loss.backward()
            
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Validation phase
        avg_train_loss = np.mean(epoch_losses)
        val_metrics = model_eval(epoch, val_loader, model, args, criterion)
        
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")
        log_metrics(f"Validation", val_metrics, args, logger)

        # Learning rate scheduling
        tuning_metric = val_metrics["micro_f1"] if args.task_type == "multilabel" else val_metrics["acc"]
        scheduler.step(tuning_metric)
        
        # Checkpointing logic
        improvement = tuning_metric > best_metric
        if improvement:
            best_metric = tuning_metric
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": no_improve_count,
                "best_metric": best_metric,
            },
            improvement,
            args.savedir,
        )
        
        # Early stopping
        if no_improve_count >= args.patience:
            logger.info(f"No improvement for {args.patience} epochs. Stopping early.")
            break

    # Final evaluation on test sets
    logger.info("Loading best model for final evaluation")
    model_path = os.path.join(args.savedir, "model_best.pt")
    load_checkpoint(model, model_path)
    model.eval()
    
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
        log_metrics(f"Test Set: {test_name}", test_metrics, args, logger)


def cli_main() -> None:
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Multimodal Model Trainer")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert not remaining_args, f"Unrecognized arguments: {remaining_args}"
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
