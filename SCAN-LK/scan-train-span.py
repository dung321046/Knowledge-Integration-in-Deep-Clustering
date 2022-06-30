import argparse
import os

import numpy as np
import torch
from termcolor import colored

from utils.common_config import get_train_transformations, get_val_transformations, \
    get_train_dataset, get_val_dataloader, get_val_dataset_henry, \
    get_optimizer, get_model, get_criterion, get_train_dataloader, \
    adjust_learning_rate
from utils.config import create_config
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from utils.train_span import scan_train

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')


def produce_clustering(model, model_path, val_dataloader, p):
    model_checkpoint = torch.load(model_path, map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)[0]
    return predictions


PATH = "SCAN-LK\\"


def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    from data.stl import STL10
    dataset = STL10(transform=val_transformations, split='train', download=True)
    val_dataset = get_val_dataset_henry(dataset, p['topk_neighbors_train_path'], to_neighbors_dataset=True)
    val_dataloader = get_val_dataloader(p, val_dataset)
    m = 1000
    # (airplane, bird), (cat, dog),  (deer, horse)
    span_targets = [[0, 1], [3, 5], [4, 6]]
    span_groups = []
    for span_target in span_targets:
        span_groups.append({"group": [], "target": span_target})

    for i, y in enumerate(dataset.targets):
        for group, span_target in enumerate(span_targets):
            if y in span_target and len(span_groups[group]["group"]) < m:
                span_groups[group]["group"].append(i)
    train_dataset = get_train_dataset(p, train_transformations, split='train', to_neighbors_dataset=True,
                                      pw_dict=None, span=span_groups)
    train_dataloader = get_train_dataloader(p, train_dataset)
    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)
    model = get_model(p, p['pretext_model'])
    # print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    # Checkpoint
    print(p['scan_checkpoint'])
    if os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
    else:
        checkpoint = torch.load(
            r"SCAN-LK\results"'\\' + p[
                'train_db_name'] + r"\scan-baseline\checkpoint.pth.tar", map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None

    # Main loop
    print(colored('Starting main loop', 'blue'))
    sdd_root = dict()
    from sdd_clustering.convert_to_T import load_span_sdd
    for i in range(10):
        for j in range(i + 1, 10):
            sdd_root[(i, j)] = load_span_sdd("B", i, j, 10)
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)
        np_predictions = np.asarray(predictions[0]["predictions"])
        for group in range(3):
            from collections import Counter
            c = Counter([np_predictions[g] for g in span_groups[group]["group"]])
            topC = c.most_common(2)
            for g in span_groups[group]["group"]:
                train_dataloader.dataset.major1[g] = topC[0][0]
                train_dataloader.dataset.major2[g] = topC[1][0]
        # train_dataloader.

        # Train
        print('Train ...')
        scan_train(train_dataloader, model, criterion, optimizer, epoch,
                   p['update_cluster_head_only'], sdd_root)

        print('Evaluate based on SCAN loss ...')
        scan_stats = scan_evaluate(predictions, None)
        print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']

        if lowest_loss < best_loss:
            print('New lowest loss on validation set: %.4f -> %.4f' % (best_loss, lowest_loss))
            print('Lowest loss head is %d' % (lowest_loss_head))
            best_loss = lowest_loss
            best_loss_head = lowest_loss_head
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'])
        else:
            print('No new lowest loss on validation set: %.4f -> %.4f' % (best_loss, lowest_loss))
            print('Lowest loss head is %d' % (best_loss_head))

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head},
                   p['scan_checkpoint'])

    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions,
                                          class_names=val_dataset.dataset.classes,
                                          compute_confusion_matrix=True,
                                          confusion_matrix_file=os.path.join(p['scan_dir'], 'confusion_matrix.png'))
    print(clustering_stats)


if __name__ == "__main__":
    main()
