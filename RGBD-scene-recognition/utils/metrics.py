import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

def calc_metrics_for_CPM(predict, softmax, logit, label):
    label_onehot = torch.nn.functional.one_hot(torch.from_numpy(label))
    label_onehot = label_onehot.squeeze().cpu().detach().numpy()
    correct = (predict==label)
    acc = np.count_nonzero(correct)/len(predict)
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    aupr, fpr = calc_fpr_aupr(softmax, correct)
    ece = calc_ece(softmax, label, bins=15)
    nll, brier = calc_nll_brier(softmax, logit, label, label_onehot)
    
    return acc, aurc, eaurc, aupr, fpr, ece, nll, brier
    
def calc_metrics_new(loader, model, criterion, t):

    acc, softmax, correct, logit = get_metric_values(loader, model, criterion, t)
    # aurc, eaurc
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    # fpr, aupr
    aupr, fpr = calc_fpr_aupr(softmax, correct)
    # calibration measure ece , mce, rmsce
    ece = calc_ece(softmax, label, bins=15)
    # brier, nll
    nll, brier = calc_nll_brier(softmax, logit, label, label_onehot)

    return acc, aurc, eaurc, aupr, fpr, ece, nll, brier    


def calc_metrics(loader, label, model, criterion, t):
    label = np.array(label)
    label_onehot = torch.nn.functional.one_hot(torch.from_numpy(label))
    label_onehot = label_onehot.squeeze().cpu().detach().numpy()
    acc, softmax, correct, logit = get_metric_values(loader, model, criterion, t)
    # aurc, eaurc
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    # fpr, aupr
    aupr, fpr = calc_fpr_aupr(softmax, correct)
    # calibration measure ece , mce, rmsce
    ece = calc_ece(softmax, label, bins=15)
    # brier, nll
    nll, brier = calc_nll_brier(softmax, logit, label, label_onehot)

    return acc, aurc, eaurc, aupr, fpr, ece, nll, brier


# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)
    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    aupr_err = metrics.average_precision_score(correctness, softmax_max)

    #print("AUPR {0:.2f}".format(aupr_err*100))
    #print('FPR {0:.2f}'.format(fpr_in_tpr_95*100))

    return aupr_err, fpr_in_tpr_95

# ECE
def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    #print("ECE {0:.2f} ".format(ece.item()*100))

    return ece.item()

# NLL & Brier Score
def calc_nll_brier(softmax, logit, label, label_onehot):
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label.long())

    #print("NLL {0:.2f} ".format(nll.item()*10))
    #print('Brier {0:.2f}'.format(brier_score*100))

    return nll.item(), brier_score

# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)

# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    #print("AURC {0:.2f}".format(aurc*1000))
    #print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc

# Get softmax, logit
def get_metric_values(loader, model, criterion, t):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        accuracy = 0

        list_softmax = []
        list_correct = []
        list_logit = []

        for input, target, idx in loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            output = output/t

            loss = criterion(output, target.long()).cuda()

            total_loss += loss.mean().item()
            pred = output.data.max(1, keepdim=True)[1]

            total_acc += pred.eq(target.data.view_as(pred)).sum()

            for i in output:
                list_logit.append(i.cpu().data.numpy())

            list_softmax.extend(F.softmax(output).cpu().data.numpy())

            for j in range(len(pred)):
                if pred[j] == target[j]:
                    accuracy += 1
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)

        total_loss /= len(loader)
        total_acc = 100. * total_acc / len(loader.dataset)

        #print('Accuracy {:.2f}'.format(total_acc))

    return total_acc.item(), list_softmax, list_correct, list_logit
