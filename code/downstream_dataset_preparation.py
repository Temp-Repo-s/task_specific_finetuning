
import csv
from glob import glob
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

def prepare_wsss4luad_dataset():
    wsss4luad_root = "data/WSSS4LUAD/training"  # 10091 files
    wsss4luad_files = os.listdir(wsss4luad_root)
    wsss4luad_files = [f for f in wsss4luad_files if (f.__contains__('[1, 0, 0]') or f.__contains__('[0, 1, 0]') or f.__contains__('[0, 0, 1]'))]  # 4693 files
    
    tumor = [f for f in wsss4luad_files if (f.__contains__('[1, 0, 0]'))]
    stroma = [f for f in wsss4luad_files if (f.__contains__('[0, 1, 0]'))]
    normal = [f for f in wsss4luad_files if (f.__contains__('[0, 0, 1]'))]
    print(f"tumor: {len(tumor)}, stroma: {len(stroma)}, normal: {len(normal)}")  # tumor: 1181, stroma: 1680, normal: 1832
    output_csv = wsss4luad_root.replace("training", "single_label.csv")

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Label"])  # CSV header

        for label, cls in enumerate([normal, stroma, tumor]):
            for filename in cls:
                writer.writerow([filename, label])


def create_wsss4luad_dataset():
    wsss4luad_root = "data/WSSS4LUAD/training" 
    ds_csv = wsss4luad_root.replace("training", "single_label.csv")  # 4693 files
    df = pd.read_csv(ds_csv)
    filenames = df["Filename"].tolist()
    check_filename = filenames[0].__contains__("/")
    if not check_filename:
        filenames = [wsss4luad_root+"/"+f for f in filenames]
    labels = df["Label"].tolist()
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels


def create_BCAH_dataset():
    idx_to_class = get_idx_to_class("BACH")
    class_to_idx = {v:k for k, v in idx_to_class.items()}

    BCAH_root = "data/BACH/Photos"
    ds_csv = BCAH_root.replace("Photos", "microscopy_ground_truth.csv")  # 400 files
    df = pd.read_csv(ds_csv, header=None)
    filenames = df[0].tolist()
    if not filenames[0].__contains__("/"):
        filenames = [BCAH_root+"/"+f for f in filenames]
    labels = df[1].tolist()
    labels = [class_to_idx[cls] for cls in labels]
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels

def prepare_SICAP_dataset():
    SICAP_root = "data/SICAPv2"

    df_final = pd.DataFrame(columns=["image_name", "label", "IsTest"])
    label_csv = f"{SICAP_root}/image_label.csv"
    for set in ["Train", "Test"]:
        df = pd.read_excel(os.path.join(
            SICAP_root, "partition", "Test", f"{set}.xlsx"))   # 9959 train, 2122 test
        df["label"] = -1
        df["IsTest"] = 1 if set == "Test" else 0
        for idx, cls in enumerate(["NC", "G3", "G4", "G5"]):
            df.loc[df[cls] == 1, "label"] = idx
        df.drop(columns=["NC", "G3", "G4", "G5", "G4C"], inplace=True)
        print(f"len(df) = {len(df)}")
        df_final = pd.concat([df_final, df])
        print(f"len(df_final) = {len(df_final)}")
        print(np.unique(df["label"].to_numpy(), return_counts=True))
    df_final.to_csv(label_csv, index=False)

def create_SICAP_dataset(set="test"):
    assert set in ["train", "test"]
    SICAP_root = "data/SICAPv2"
    ds_csv = os.path.join(SICAP_root, "image_label.csv")
    df = pd.read_csv(ds_csv)
    if set == "test":
        df = df[df["IsTest"] == 1]
    else:
        df = df[df["IsTest"] == 0]
    filenames = df["image_name"].tolist()
    if not filenames[0].__contains__("/"):
        filenames = [SICAP_root+"/images/"+f for f in filenames]
    # check_existence = [f for f in filenames if not os.path.exists(f)]
    # print(f"the following files do not exist: {check_existence}")
    labels = df["label"].tolist()
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels

def create_MHIST_dataset(set="test"):
    assert set in ["train", "test"]
    MHIST_root = "data/MHIST"
    ds_csv = os.path.join(MHIST_root, "annotations.csv")
    df = pd.read_csv(ds_csv)
    df = df[df["Partition"] == set]
    filenames = df["Image Name"].tolist()
    if not filenames[0].__contains__("/"):
        filenames = [MHIST_root+"/images/"+f for f in filenames]
    # check_existence = [f for f in filenames if not os.path.exists(f)]
    # print(f"the following files do not exist: {check_existence}")
    labels = df["Majority Vote Label"].tolist()
    labels = [1 if ele == "SSA" else 0 for ele in labels]
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels


def five_fold_split(filenames, labels, five_fold_split_csv):
    df = pd.DataFrame({"Filename": filenames,
                       "Label": labels})
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["Fold"] = -1
    for fold, (train_idx, test_idx) in enumerate(skf.split(df["Filename"], df["Label"])):
        df.loc[test_idx, 'Fold'] = fold
    df.to_csv(five_fold_split_csv, index=False)


def create_BACH_five_fold_split():
    filenames, labels = get_task_images_labels(task="BACH")
    five_fold_split_csv = "CONCH/results/BACH_5_folds.csv"
    five_fold_split(filenames, labels, five_fold_split_csv)
    
def create_wsss4luad_five_fold_split():
    filenames, labels = get_task_images_labels(task="wsss4luad")
    five_fold_split_csv = "CONCH/results/wsss4luad_5_folds.csv"

    WSI_ids = list(set([f.rsplit("-", 3)[0] for f in filenames]))
    kf = KFold(n_splits=5)
    WSI_folds = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(WSI_ids)):
        for idx in test_idx:
            WSI_folds[WSI_ids[idx]] = fold

    data = []
    for filename, label in zip(filenames, labels):
        WSI_id = filename.rsplit("-", 3)[0]
        fold = WSI_folds[WSI_id]
        data.append((filename, label, fold))

    df = pd.DataFrame(data, columns=["Filename", "Label", "Fold"])
    df.to_csv(five_fold_split_csv, index=False)

    # [np.unique(df[df["Fold"] == fold]["Label"].to_numpy(), return_counts=True) for fold in range(5)]
    """
    0 = (array([0, 1, 2]), array([406, 317, 178]))
    1 = (array([0, 1, 2]), array([205, 498, 286]))
    2 = (array([0, 1, 2]), array([598, 289, 390]))
    3 = (array([0, 1, 2]), array([376, 291, 184]))
    4 = (array([0, 1, 2]), array([247, 285, 143]))
    """


def create_MHIST_five_fold_split():
    filenames, labels = create_MHIST_dataset(set="train")
    five_fold_split_csv = "CONCH/results/MHIST_5_folds.csv"
    five_fold_split(filenames, labels, five_fold_split_csv)


def create_SICAP_five_fold_split():
    filenames, labels = create_SICAP_dataset(set="train")
    five_fold_split_csv = "CONCH/results/SICAP_5_folds.csv"

    WSI_ids = list(set([f.rsplit("_Block_Region_", 1)[0] for f in filenames]))
    kf = KFold(n_splits=5)
    WSI_folds = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(WSI_ids)):
        for idx in test_idx:
            WSI_folds[WSI_ids[idx]] = fold

    data = []
    for filename, label in zip(filenames, labels):
        WSI_id = filename.rsplit("_Block_Region_", 1)[0]
        fold = WSI_folds[WSI_id]
        data.append((filename, label, fold))

    df = pd.DataFrame(data, columns=["Filename", "Label", "Fold"])
    df.to_csv(five_fold_split_csv, index=False)

    # [np.unique(df[df["Fold"] == fold]["Label"].to_numpy(), return_counts=True) for fold in range(5)]
    """
    0 = (array([0, 1, 2, 3]), array([347, 445, 858,  62]))
    1 = (array([0, 1, 2, 3]), array([ 635,  272, 1004,  149]))
    2 = (array([0, 1, 2, 3]), array([806, 473, 869, 102]))
    3 = (array([0, 1, 2, 3]), array([1134,  443,  317,  123]))
    4 = (array([0, 1, 2, 3]), array([851, 196, 593, 280]))
    """

def get_idx_to_class(task):
    
    if task == "wsss4luad":
        idx_to_class = {0: "normal", 1: "stroma", 2: "tumor"}
    elif task == "BACH":
        idx_to_class = {0: "Normal", 1: "Benign", 2: "InSitu", 3: "Invasive"}
    elif task == "SICAP":
        idx_to_class = {0: "NC", 1: "G3", 2: "G4", 3: "G5"}
    elif task == "MHIST":
        idx_to_class = {0: "HP", 1: "SSA"}
    else:
        raise ValueError
    return idx_to_class

def get_task_images_labels(task):
    if task == "wsss4luad":
        filenames, labels = create_wsss4luad_dataset()
    elif task == "BACH":
        filenames, labels = create_BCAH_dataset()
    elif task == "SICAP":
        filenames, labels = create_SICAP_dataset()
    elif task == "MHIST":
        filenames, labels = create_MHIST_dataset()
    else:
        raise ValueError
    
    return filenames, labels


def get_task_organ(task):
    if task == "wsss4luad":
        organ = ["lung", "pulmonary"]
    elif task == "BACH":
        organ = ["breast"]
    elif task == "SICAP":
        organ = ["prostate"]
    elif task == "MHIST":
        organ = ["colon", "colorectal", "polyp"]
    else:
        raise ValueError
    return organ


def get_keyword(task):
    organ = get_task_organ(task)
    if task == "wsss4luad":
        class_keys = ["epithelial",
                      "stroma",
                      " normal "]
    elif task == "BACH":
        class_keys = ["normal", "benign", "in situ", "invasive"]
    elif task == "SICAP":
        class_keys = ['grade', 'gleason', 'benign']
    elif task == "MHIST":
        class_keys = ["hyperplastic", "benign", "sessile", "serrated", "adenoma" , "precancerous"]
    else:
        raise ValueError
    
    return organ, class_keys


# create_BACH_five_fold_split()
# create_wsss4luad_five_fold_split()
# create_MHIST_five_fold_split()
# create_SICAP_five_fold_split()
