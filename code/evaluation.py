import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             classification_report, roc_auc_score)
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot
from conch.downstream.utils import AverageMeter

import json
from pathlib import Path
import os

from downstream_dataset_preparation import *
from utils import *


def zero_shot_conch_eval(task, model, preprocess):
    filenames, labels = None, None
    idx_to_class = get_idx_to_class(task)
    prompt_file = f'CONCH/prompts/{task}_prompts_all_per_class.json'
    filenames, labels = get_task_images_labels(task)
    dataset = CustomImageDataset(filenames, labels, preprocess)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    print("num samples: ", len(dataloader.dataset))

    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])]
                       for idx in range(n_classes)]
    for class_idx, classname in enumerate(classnames_text):
        print(f'{class_idx}: {classname}')

    _ = model.eval()
    zeroshot_weights = zero_shot_classifier(
        model, classnames_text, templates, device=device)
    print(zeroshot_weights.shape)

    results, _ = run_zeroshot(model, zeroshot_weights,
                              dataloader, device, dump_results=False)
    print(results)
    return results


def supervised_eval(task, linear_probing_model, logit_scale, preprocess, five_fold_split_csv, fold,
                    metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1']):
    if task in ["BACH", "wsss4luad"]:
        test_image_paths, test_labels = get_train_val_split(five_fold_split_csv, folds=[fold])
    elif task in ["MHIST", "SICAP"]:
        test_image_paths, test_labels = get_task_images_labels(task)
    else:
        raise ValueError
    print(f"{len(test_image_paths)} test_image_paths loaded")
    print(test_image_paths[:5])
    print(test_labels[:5])
    test_dataset = CustomImageDataset(test_image_paths, test_labels,
                                      transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=min(len(test_dataset), 8), shuffle=False,
                             num_workers=4, pin_memory=True, drop_last=False)

    linear_probing_model = linear_probing_model.to(device)
    linear_probing_model.eval()

    acc_meter = AverageMeter()
    logits_all, targets_all, preds_all = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            batch_size = targets.size(0)

            logits = linear_probing_model(images)
            preds = logits.argmax(dim=1)

            logits_all.append(logits.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
            preds_all.append(preds.cpu().numpy())

            # Update AverageMeters with new results
            acc_meter.update(
                (preds == targets).float().mean().item(), n=batch_size)

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = F.softmax(torch.from_numpy(logits_all) *
                          logit_scale.exp().item(), dim=1).numpy()
    preds_all = np.concatenate(preds_all, axis=0)
    bacc = balanced_accuracy_score(targets_all, preds_all)
    weighted_kappa = cohen_kappa_score(
        targets_all, preds_all, weights='quadratic')
    kappa = cohen_kappa_score(targets_all, preds_all)
    cls_rep = classification_report(
        targets_all, preds_all, output_dict=True, zero_division=0)
    acc = acc_meter.avg

    n_classes = probs_all.shape[1]
    if n_classes == 2:
        class_probs = probs_all[:, 1]
        roc_kwargs = {}
    else:
        class_probs = probs_all
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    try:
        roc_auc = roc_auc_score(targets_all, class_probs, **roc_kwargs)
    except ValueError:
        roc_auc = np.nan

    results = {'acc': acc,
               'bacc': bacc,
               'weighted_kappa': weighted_kappa,
               'kappa': kappa,
               'roc_auc': roc_auc,
               'weighted_f1': cls_rep['weighted avg']['f1-score']}
    results = {k: results[k] for k in metrics}
    print(results)

    return results


@torch.no_grad()
def zero_shot_clip_classifier(model, classnames, templates, tokenizer, device=None):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """
    zeroshot_weights = []
    for classnames_for_class in classnames:
        embeddings_for_class = []
        for classname in classnames_for_class:
            texts = [template.replace('CLASSNAME', classname)
                     for template in templates]

            texts = tokenizer(texts).to(device)
            classname_embeddings = model.encode_text(texts, normalize=True)
            # classname_embeddings: [num_templates, embedding_dim]
            embeddings_for_class.append(
                F.normalize(classname_embeddings, dim=-1))

        # class_embedding: [num_classnames, num_templates, embedding_dim]
        class_embedding = torch.stack(embeddings_for_class, dim=0)
        # over all templates and classnames
        class_embedding = class_embedding.mean(dim=(0, 1))
        class_embedding /= class_embedding.norm()

        # class_embedding: [embedding_dim]
        zeroshot_weights.append(class_embedding)

    # zeroshot_weights: [embedding_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def zero_shot_clip_eval(task, model, preprocess, tokenizer):
    filenames, labels = None, None
    idx_to_class = get_idx_to_class(task)
    prompt_file = f'CONCH/prompts/{task}_prompts_all_per_class.json'
    filenames, labels = get_task_images_labels(task)
    dataset = CustomImageDataset(filenames, labels, preprocess)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    print("num samples: ", len(dataloader.dataset))

    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])]
                       for idx in range(n_classes)]
    for class_idx, classname in enumerate(classnames_text):
        print(f'{class_idx}: {classname}')

    _ = model.eval()
    zeroshot_weights = zero_shot_clip_classifier(
        model, classnames_text, templates, tokenizer, device=device)
    print(zeroshot_weights.shape)

    results, _ = run_zeroshot(model, zeroshot_weights,
                              dataloader, device, dump_results=False)
    print(results)
    return results


def zero_shot_eval(vl_model, task, model, preprocess, tokenizer=None):
    model.eval()
    model = model.to(device)
    if vl_model == "conch":
        return zero_shot_conch_eval(task, model, preprocess)
    elif vl_model == "quiltNet":
        return zero_shot_clip_eval(task, model, preprocess, tokenizer)
    else:
        raise ValueError
