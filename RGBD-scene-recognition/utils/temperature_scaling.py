import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.depth_rgb_temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.rgb_temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.depth_temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, rgb, depth):
        depth_rgb_logits, rgb_logits, depth_logits = self.model(rgb, depth)

        return self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)

    def temperature_scale(self, depth_rgb_logits, rgb_logits, depth_logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        depth_rgb_temperature = self.depth_rgb_temperature.unsqueeze(1).expand(depth_rgb_logits.size(0), depth_rgb_logits.size(1))
        rgb_temperature = self.rgb_temperature.unsqueeze(1).expand(rgb_logits.size(0), rgb_logits.size(1))
        depth_temperature = self.depth_temperature.unsqueeze(1).expand(depth_logits.size(0), depth_logits.size(1))

        return depth_rgb_logits / depth_rgb_temperature, rgb_logits / rgb_temperature, depth_logits / depth_temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        depth_rgb_logits_list, rgb_logits_list, depth_logits_list = [], [], []
        labels_list = []
        with torch.no_grad():
            for batch in valid_loader:
                rgb, depth, label = batch['A'], batch['B'], batch['label']
                rgb, depth, label = rgb.cuda(), depth.cuda(), label.cuda()
                depth_rgb_logits, rgb_logits, depth_logits = self.model(rgb, depth)
                depth_rgb_logits_list.append(depth_rgb_logits)
                rgb_logits_list.append(rgb_logits)
                depth_logits_list.append(depth_logits)
                labels_list.append(label)
            depth_rgb_logits = torch.cat(depth_rgb_logits_list).cuda()
            rgb_logits = torch.cat(rgb_logits_list).cuda()
            depth_logits = torch.cat(depth_logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(depth_rgb_logits, labels).item()
        before_temperature_ece = ece_criterion(depth_rgb_logits, labels).item()
        print('FULL Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        before_temperature_nll = nll_criterion(rgb_logits, labels).item()
        before_temperature_ece = ece_criterion(rgb_logits, labels).item()
        print('RGB Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        before_temperature_nll = nll_criterion(depth_logits, labels).item()
        before_temperature_ece = ece_criterion(depth_logits, labels).item()
        print('DEPTH Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        

        # Next: optimize the temperature w.r.t. NLL
        depth_rgb_optimizer = optim.LBFGS([self.depth_rgb_temperature], lr=0.01, max_iter=50)
        rgb_optimizer = optim.LBFGS([self.rgb_temperature], lr=0.01, max_iter=50)
        depth_optimizer = optim.LBFGS([self.depth_temperature], lr=0.01, max_iter=50)

        def depth_rgb_eval():
            depth_rgb_optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[0], labels)
            loss.backward()
            return loss
        depth_rgb_optimizer.step(depth_rgb_eval)
        def rgb_eval():
            rgb_optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[1], labels)
            loss.backward()
            return loss
        rgb_optimizer.step(rgb_eval)
        def depth_eval():
            depth_optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[2], labels)
            loss.backward()
            return loss
        depth_optimizer.step(depth_eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[0], labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[0], labels).item()
        print('FULL Optimal temperature: %.3f' % self.depth_rgb_temperature.item())
        print('FULL After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        after_temperature_nll = nll_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[1], labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[1], labels).item()
        print('RGB Optimal temperature: %.3f' % self.rgb_temperature.item())
        print('RGB After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        after_temperature_nll = nll_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[2], labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(depth_rgb_logits, rgb_logits, depth_logits)[2], labels).item()
        print('DEPTH Optimal temperature: %.3f' % self.depth_temperature.item())
        print('DEPTH After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
