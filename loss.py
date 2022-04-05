######### Loss function ############

import torch


def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)
    
    intersection = torch.stack(
        [torch.sum((preds == i) & (masks == i), dim=(1, 2)) for i in range(num_classes)],
        dim=1
    )
    union = torch.stack(
        [torch.sum((preds == i) | (masks == i), dim=(1, 2)) for i in range(num_classes)],
        dim=1
    )
    target = torch.stack(
        [torch.sum(masks == i, dim=(1, 2)) for i in range(num_classes)],
        dim=1
    )
    # Output shapes: B x num_classes

    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    mean_iou = torch.mean(intersection / (union + eps), dim=0)
    mean_class_acc = torch.mean(intersection / (target + eps), dim=0) # TODO: calc mean class accuracy
    mean_acc = torch.mean(intersection / (target + eps), dim=(0,1)) # TODO: calc mean accuracy

    return mean_iou, mean_class_acc, mean_acc