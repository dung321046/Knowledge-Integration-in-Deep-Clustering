import numpy as np
import torch
import torch.nn as nn

from sdd_clustering.calculate_wmc import ProbCalculator, weight_convert_b
from utils.utils import AverageMeter, ProgressMeter

softmax = nn.Softmax(dim=1)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False, sdd_root=None):
    """
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    expert_losses = AverageMeter('Expert', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, consistency_losses, entropy_losses, expert_losses],
                             prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN
    wmc = 0.0
    nclauses = 0.0000000000001
    nsat = 0
    for batch_id, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)
        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                      neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))
        expert_loss = 0.0
        vbatch = anchors_output[0]
        b, n = vbatch.size()

        pbatch = softmax(vbatch)
        for i in range(b):
            mc1, mc2 = batch["major1"][i].item(), batch["major2"][i].item()
            if batch["major1"][i] != -1:
                # value = pbatch[i][mc1] + pbatch[i][mc2]
                prob_cal = ProbCalculator(weight_convert_b([pbatch[i]], 1, 10))
                root = sdd_root[(min(mc1, mc2), max(mc1, mc2))]
                value = prob_cal.calculate(root)
                wmc += value.cpu().detach().numpy()
                expert_loss += -0.1 * torch.log(value)
                if torch.argmax(pbatch[i]) == mc1 or torch.argmax(pbatch[i]) == mc2:
                    nsat += 1
                nclauses += 1
        expert_losses.update(expert_loss)
        total_loss += expert_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if batch_id % 25 == 0:
            progress.display(batch_id)
    print("WMC:", wmc / nclauses, "#Constraints:", nclauses, "#Sat", nsat)
