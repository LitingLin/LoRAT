from torch import nn

from trackit.miscellanies.pretty_format import pretty_format


def build_box_with_score_map_criteria(criteria_config: dict):
    print('criteria config:\n' + pretty_format(criteria_config))
    classification_config = criteria_config['classification']
    if classification_config['type'] == 'binary_cross_entropy':
        cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        cls_loss_name = 'bce'
    elif classification_config['type'] == 'varifocal':
        from ...modules.varifocal_loss import VarifocalLoss
        cls_loss = VarifocalLoss(alpha=classification_config['alpha'],
                                 gamma=classification_config['gamma'],
                                 iou_weighted=classification_config['iou_weighted'])
        cls_loss_name = 'varifocal'
    else:
        raise NotImplementedError(f"Classification type {classification_config['type']} is not implemented")
    iou_aware_classification_score = classification_config['iou_aware_classification_score']
    cls_loss_weight = classification_config['weight']

    bbox_regression_config = criteria_config['bbox_regression']
    if bbox_regression_config['type'] == 'iou':
        from ...modules.iou_loss import IoULoss
        bbox_reg_loss = IoULoss()
        bbox_reg_loss_name = 'iou'
    elif bbox_regression_config['type'] == 'GIoU':
        from ...modules.iou_loss import GIoULoss
        bbox_reg_loss = GIoULoss()
        bbox_reg_loss_name = 'iou'
    else:
        raise NotImplementedError(f"BBox regression type {bbox_regression_config['type']} is not implemented")
    bbox_reg_loss_weight = bbox_regression_config['weight']

    from . import SimpleCriteria
    return SimpleCriteria(cls_loss, bbox_reg_loss, iou_aware_classification_score,
                          cls_loss_weight, bbox_reg_loss_weight, cls_loss_name, bbox_reg_loss_name)
