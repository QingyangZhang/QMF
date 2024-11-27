import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.QMF import QMF
import torchvision.transforms as transforms
from data.aligned_conc_dataset_noised import AlignedConcDataset
from data.additional_transform import AddGaussianNoise
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
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--name", type=str, default="s")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--savedir", type=str, default="./savepath/QMF/nyud/pretrained_resnet18/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--severity", type=float, default=5.0)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str,
                        default="./checkpoint/resnet18_pretrained.pth")



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


def test(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name, str(args.seed))

    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    val_transforms = list()
    val_transforms.append(transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)))
    val_transforms.append(transforms.RandomApply([AddGaussianNoise(mean=0.0, variance=args.severity)], p=0.5))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=mean, std=std))

    test_loader = DataLoader(
        AlignedConcDataset(
            args, 
            data_dir=os.path.join(args.data_path, 'test'), 
            rgb_transform=transforms.Compose(val_transforms),
            depth_transform = transforms.Compose(val_transforms),
        ),
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers)
        
    model = QMF(args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()

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
    test(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
