import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from prettytable import PrettyTable  # 添加表格输出

from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('prediction_path', help='path to .pkl result file')
    parser.add_argument('save_dir', help='directory to save confusion matrix')
    parser.add_argument('--show', action='store_true', help='show confusion matrix')
    parser.add_argument('--color-theme', default='plasma', help='theme of the matrix color map')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold to filter bboxes')
    parser.add_argument('--tp-iou-thr', type=float, default=0.5, help='IoU threshold for true positives')
    parser.add_argument('--nms-iou-thr', type=float, default=None, help='nms IoU threshold')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    return parser.parse_args()


def calculate_confusion_matrix(dataset, results, score_thr=0, nms_iou_thr=None, tp_iou_thr=0.5):
    """Calculate the confusion matrix."""
    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results), "Dataset and results length mismatch."

    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr, tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix, gts, result, score_thr=0, tp_iou_thr=0.5, nms_iou_thr=None):
    """Analyze detection results for each image."""
    true_positives = np.zeros(len(gts))
    gt_bboxes, gt_labels = [], []

    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        if nms_iou_thr:
            det_bboxes, _ = nms(det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)

        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # True positive
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # False positive
                    confusion_matrix[-1, det_label] += 1

    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # False negative
            confusion_matrix[gt_label, -1] += 1


def calculate_metrics(confusion_matrix, labels):
    """Calculate precision, recall, Dice, and IoU for each class."""
    num_classes = len(labels) - 1
    metrics = []

    total_tp, total_fp, total_fn = 0, 0, 0

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, -1]
        fp = confusion_matrix[-1, i]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)

        metrics.append({'class': labels[i], 'precision': precision, 'recall': recall, 'dice': dice, 'iou': iou})

        total_tp += tp
        total_fp += fp
        total_fn += fn

    mean_precision = total_tp / (total_tp + total_fp + 1e-6)
    mean_recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-6)

    return metrics, mean_precision, mean_recall, mean_dice


def plot_confusion_matrix(confusion_matrix, labels, save_dir=None, show=True, color_theme='plasma'):
    """Plot the confusion matrix."""
    num_classes = len(labels)
    fig, ax = plt.subplots(figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(top=True, labeltop=True)

    fig.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    results = load(args.prediction_path)

    if not results:
        print("Error: Results file is empty or not loaded correctly.")
        return

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    confusion_matrix = calculate_confusion_matrix(dataset, results, args.score_thr, args.nms_iou_thr, args.tp_iou_thr)
    labels = dataset.metainfo['classes'] + ('background', )
    plot_confusion_matrix(confusion_matrix, labels, save_dir=args.save_dir, show=args.show)

    metrics, mean_precision, mean_recall, mean_dice = calculate_metrics(confusion_matrix, labels)

    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "Dice", "IoU"]
    for metric in metrics:
        table.add_row([
            metric['class'], f"{metric['precision']:.4f}", f"{metric['recall']:.4f}",
            f"{metric['dice']:.4f}", f"{metric['iou']:.4f}"
        ])
    print("\nPer-class Metrics:")
    print(table)

    print("\nAverage Metrics:")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall:    {mean_recall:.4f}")
    print(f"Mean Dice:      {mean_dice:.4f}")


if __name__ == '__main__':
    main()
