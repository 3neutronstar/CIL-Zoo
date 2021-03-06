import torch

def batch_calibration_stats(logits, targets, num_bins):
    bin_bounds = torch.linspace(1 / num_bins, 1.0, num_bins).to(logits.device)
    probs, preds = logits.softmax(dim=-1).max(-1)
    bin_correct = torch.zeros(num_bins).float()
    bin_prob = torch.zeros(num_bins).float()
    bin_count = torch.zeros(num_bins).float()
    for idx, conf_level in enumerate(bin_bounds):
        mask = (conf_level - 1 / num_bins < probs) * (probs <= conf_level)
        num_elements = mask.sum().float()
        total_correct = 0. if num_elements < 1 else preds[mask].eq(targets[mask]).sum()
        total_prob = 0. if num_elements < 1 else probs[mask].sum()
        bin_count[idx] = num_elements
        bin_correct[idx] = total_correct
        bin_prob[idx] = total_prob
    return bin_count, bin_correct, bin_prob


def expected_calibration_err(bin_count, bin_correct, bin_prob, num_samples):
    ece = 0
    for count, correct, prob in zip(bin_count, bin_correct, bin_prob):
        if count < 1:
            continue
        ece += count / num_samples * abs(correct / count - prob / count)
    return ece.item()

def ece_bin_metrics(bin_count, bin_correct, bin_prob, num_bins, prefix):
    bin_bounds = torch.linspace(1 / num_bins, 1.0, num_bins)
    assert bin_bounds.size(0) == bin_count.size(0)
    bin_acc = map(lambda x: 0. if x[1] < 1 else (x[0] / x[1]).item(), zip(bin_correct, bin_count))
    bin_conf = map(lambda x: 0. if x[1] < 1 else (x[0] / x[1]).item(), zip(bin_prob, bin_count))
    metrics = {f"{prefix}_bin_count_{ub:0.2f}": count.item() for ub, count in zip(bin_bounds, bin_count)}
    metrics.update(
        {f"{prefix}_bin_acc_{ub:0.2f}": acc for ub, acc in zip(bin_bounds, bin_acc)}
    )
    metrics.update(
        {f"{prefix}_bin_conf_{ub:0.2f}": conf for ub, conf in zip(bin_bounds, bin_conf)}
    )
    return metrics