import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.MMTM import MMTNet
import torchvision.transforms as transforms
from data.aligned_conc_dataset import AlignedConcDataset
from utils.utils import *
from utils.logger import create_logger, get_test_logger
from utils.metrics import calc_metrics_for_CPM
import os
from torch.utils.data import DataLoader

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="/media/zhangqingyang/dataset/mm/nyud2_trainvaltest/")
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
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="g")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/MMTM/nyud/pretrained_resnet18/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--fc_final_preds", type=bool, default=True)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str,
                        default="./checkpoint/resnet18_pretrained.pth")


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_forward(i_epoch, model, args, batch):
    rgb, depth, tgt = batch['A'], batch['B'], batch['label']

    rgb, depth, tgt = rgb.cuda(), depth.cuda(), tgt.cuda()
    #depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb, depth)
    depth_rgb_logits, rgb_logits, depth_logits = model(rgb, depth)
    
    depth_clf_loss = nn.CrossEntropyLoss()(depth_logits, tgt)
    rgb_clf_loss = nn.CrossEntropyLoss()(rgb_logits, tgt)
    depth_rgb_clf_loss = nn.CrossEntropyLoss()(depth_rgb_logits, tgt)
    clf_loss = depth_clf_loss + rgb_clf_loss + depth_rgb_clf_loss
    # clf_loss = depth_rgb_clf_loss
    
    depth_prob = F.softmax(depth_logits)
    depth_conf = torch.max(depth_prob, axis=1)[0]
    rgb_prob = F.softmax(rgb_logits)
    rgb_conf = torch.max(rgb_prob, axis=1)[0]
    depth_rgb_prob = F.softmax(depth_rgb_logits)
    depth_rgb_conf = torch.max(depth_rgb_prob, axis=1)[0]

    _, predict = torch.max(depth_rgb_logits, 1)
    _, rgb_predict = torch.max(rgb_logits, 1)
    _, depth_predict = torch.max(depth_logits, 1)

    loss = torch.mean(clf_loss)
    
    return loss, depth_logits, rgb_logits, depth_rgb_logits, tgt


def model_eval(i_epoch, data, model, args, store_preds=False):
    model.eval()
    with torch.no_grad():
        losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
        label_list, depth_rgb_pred_list, depth_rgb_logits_list, depth_rgb_softmax_list = [], [], [], []
        for batch in data:
            loss, depth_logits, rgb_logits, depth_rgb_logits, tgt = model_forward(i_epoch, model, args, batch)
            losses.append(loss.item())

            depth_rgb_prob = F.softmax(depth_rgb_logits).cpu().detach().numpy()
            depth_pred = depth_logits.argmax(dim=1).cpu().detach().numpy()
            rgb_pred = rgb_logits.argmax(dim=1).cpu().detach().numpy()
            depth_rgb_pred = depth_rgb_logits.argmax(dim=1).cpu().detach().numpy()
            

            depth_preds.append(depth_pred)
            rgb_preds.append(rgb_pred)
            depthrgb_preds.append(depth_rgb_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            depth_rgb_logits_list.extend(depth_rgb_logits.cpu().detach().numpy())
            depth_rgb_softmax_list.extend(depth_rgb_prob)
            depth_rgb_pred_list.extend(depth_rgb_pred)
            label_list.extend(tgt)

    metrics = {"loss": np.mean(losses)}

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["depth_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["depthrgb_acc"] = accuracy_score(tgts, depthrgb_preds)

    logits = np.array(depth_rgb_logits_list)
    preds = np.array(depth_rgb_pred_list)
    softmax = np.array(depth_rgb_softmax_list)
    label = np.array(label_list)
    acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics_for_CPM(preds, softmax, logits, label)
    # print('{:.4f},{:.4f},{:.4f},{:.4f}'.format(acc,aurc,eaurc,aupr))
    metrics['NLL'] = nll
    metrics['AURC'] = aurc
    metrics['EAURC'] = eaurc
    metrics['AUPR'] = aupr
    metrics['FPR'] = fpr

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
    #train_transforms.append(transforms.CenterCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    #train_transforms.append(transforms.RandomVerticalFlip(0.2))
    #train_transforms.append(transforms.RandomRotation(30))
    #train_transforms.append(transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=mean, std=std))
    val_transforms = list()
    #val_transforms.append(transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)))
    val_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    # add noise
    # val_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # val_transforms.append(transforms.CenterCrop((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=mean, std=std))

    train_loader = DataLoader(
        AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'train'), transform=transforms.Compose(train_transforms)),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers)
    test_loader = DataLoader(
            AlignedConcDataset(args, data_dir=os.path.join(args.data_path, 'test'), transform=transforms.Compose(val_transforms)),
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers)
    model = MMTNet(args)
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

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, depth_out, rgb_out, depthrgb, tgt = model_forward(i_epoch, model, args, batch)
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
