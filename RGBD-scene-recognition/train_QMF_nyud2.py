import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.QMF import QMF
import torchvision.transforms as transforms
from data.aligned_conc_dataset import AlignedConcDataset
from utils.utils import *
from utils.crl_utils import *
from utils.logger import create_logger
import os
from torch.utils.data import DataLoader


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="./dataset/nyud2_trainvaltest")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--name", type=str, default="s")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/QMF/nyud/pretrained_resnet18/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str,
                        default="./checkpoint/resnet18_pretrained.pth")


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def rank_loss(confidence, idx, history):
    confidence = confidence.squeeze()
    assert len(confidence.shape) == 1
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                        rank_input2,
                                        -rank_target)

    return ranking_loss


def model_forward_train(i_epoch, model, args, batch, depth_history, rgb_history):

    rgb, depth, tgt = batch['A'], batch['B'], batch['label']
    idx = batch['idx']

    rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
    depth_rgb_logits, rgb_logits, depth_logits, rgb_conf, depth_conf = model(rgb, depth)

    depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
    rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
    depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
    clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss

    depth_pred = depth_logits.argmax(dim=1)
    rgb_pred = rgb_logits.argmax(dim=1)

    depth_correctness = (depth_pred == tgt)
    rgb_correctness = (rgb_pred == tgt)
    depth_loss = nn.CrossEntropyLoss(reduction='none')(depth_logits, tgt).detach()
    rgb_loss = nn.CrossEntropyLoss(reduction='none')(rgb_logits, tgt).detach()

    depth_rank_loss = rank_loss(depth_conf, idx, depth_history)
    rgb_rank_loss = rank_loss(rgb_conf, idx, rgb_history)


    depth_history.correctness_update(idx, depth_loss, depth_conf.squeeze())
    rgb_history.correctness_update(idx, rgb_loss, rgb_conf.squeeze())

    loss = clf_loss + args.lamb*(depth_rank_loss+rgb_rank_loss)
    
    return loss, depth_rgb_logits, rgb_logits, depth_logits, tgt

def model_forward_eval(i_epoch, model, args, batch):
    rgb, depth, tgt = batch['A'], batch['B'], batch['label']

    rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
    depth_rgb_logits, rgb_logits, depth_logits, rgb_conf, depth_conf = model(rgb, depth)
    
    depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
    rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
    depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
    clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss
    
    loss = torch.mean(clf_loss)
    
    return loss, depth_rgb_logits, rgb_logits, depth_logits, tgt


def model_eval(i_epoch, data, model, args, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
        for batch in data:
            loss, depth_rgb_logits, rgb_logits, depth_logits, tgt = model_forward_eval(i_epoch, model, args, batch)
            losses.append(loss.item())

            depth_pred = depth_logits.argmax(dim=1).cpu().detach().numpy()
            rgb_pred = rgb_logits.argmax(dim=1).cpu().detach().numpy()
            depth_rgb_pred = depth_rgb_logits.argmax(dim=1).cpu().detach().numpy()

            depth_preds.append(depth_pred)
            rgb_preds.append(rgb_pred)
            depthrgb_preds.append(depth_rgb_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgb_preds)
    return metrics


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name, str(args.seed))
    os.makedirs(args.savedir, exist_ok=True)

    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    train_transforms = list()
    train_transforms.append(transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)))
    train_transforms.append(transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=mean, std=std))

    val_transforms = list()
    val_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=mean, std=std))

    train_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'), 
        transform=transforms.Compose(train_transforms)),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers)

    test_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'), 
        transform=transforms.Compose(val_transforms)),
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers)
        
    model = QMF(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    '''
    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    '''
    
    depth_history = History(len(train_loader.dataset))
    rgb_history = History(len(train_loader.dataset))

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, depth_out, rgb_out, depthrgb, tgt = model_forward_train(i_epoch, model, args, batch, depth_history, rgb_history)
            if args.gradient_accumulation_steps > 1:
                 loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(
            np.inf, test_loader, model, args, store_preds=True
        )
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("val", metrics, logger)
        logger.info(
            "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
                "val", metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"], metrics["depthrgb_acc"]
            )
        )
        tuning_metric = metrics["depthrgb_acc"]

        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, test_loader, model, args, store_preds=True
    )
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
            "Test", test_metrics["loss"], test_metrics["depth_acc"], test_metrics["rgb_acc"],
            test_metrics["depthrgb_acc"]
        )
    )
    log_metrics(f"Test", test_metrics, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
