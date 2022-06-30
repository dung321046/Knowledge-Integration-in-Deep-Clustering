import numpy as np
import torch

from sdd_clustering.calculate_wmc import ProbCalculator, weight_convert_b
from sdd_clustering.convert_to_T import load_pw_sdd
from utils.utils import AverageMeter, ProgressMeter


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
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
    ml_root, cl_root = load_pw_sdd("B", 10)
    for batch_id, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        mls = batch['ml-image'].cuda(non_blocking=True)
        cls = batch['cl-image'].cuda(non_blocking=True)
        if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)
        mls_output = model(mls)[0]
        cls_output = model(cls)[0]
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
        if epoch > -1:
            vbatch = anchors_output[0]
            b, n = vbatch.size()

            import torch.nn as nn
            softmax = nn.Softmax(dim=1)
            pbatch = softmax(vbatch)
            mlpbatch = softmax(mls_output)
            clpbatch = softmax(cls_output)
            for i in range(b):
                if batch["ml"][i] != -1:
                    # value = torch.sum(pbatch[i] * mlpbatch[i])
                    group = [pbatch[i], mlpbatch[i]]
                    prob_cal = ProbCalculator(weight_convert_b(group, 2, 10))
                    value = prob_cal.calculate(ml_root)
                    if torch.argmax(pbatch[i]) == torch.argmax(mlpbatch[i]):
                        nsat += 1
                    expert_loss += - 0.1 * torch.log(value)
                    if type(value) != float:
                        wmc += value.cpu().detach().numpy()
                    nclauses += 1
                if batch["cl"][i] != -1:
                    #value = 1 - torch.sum(pbatch[i] * clpbatch[i])
                    group = [pbatch[i], clpbatch[i]]
                    prob_cal = ProbCalculator(weight_convert_b(group, 2, 10))
                    value = prob_cal.calculate(cl_root)
                    if torch.argmax(pbatch[i]) != torch.argmax(clpbatch[i]):
                        nsat += 1
                    expert_loss += -0.1 * torch.log(value)
                    if type(value) != float:
                        wmc += value.cpu().detach().numpy()
                    nclauses += 1
        expert_losses.update(expert_loss)
        total_loss += expert_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if batch_id % 25 == 0:
            progress.display(batch_id)
    print("WMC:", wmc / nclauses, "#Constraints:", nclauses, "#Sat", nsat)
