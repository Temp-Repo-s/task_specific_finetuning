import csv
import numpy as np
import os
from finetune import *
from linear_probing import *
import time
from utils import *

import argparse


def main():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--vl_model", type=str, default="conch")
    parser.add_argument("--run_modes", nargs="+", type=int, default=1)
    parser.add_argument("--task", type=str, default="BACH")
    parser.add_argument("--image_caption_source",
                        nargs="+", type=str, default="ARCH")
    parser.add_argument("--training_images", nargs="+", type=int, default=0)
    parser.add_argument("--subset_size", nargs="+", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)

    args = parser.parse_args()

    vl_model = args.vl_model
    task = args.task
    image_caption_source_list = args.image_caption_source
    if not isinstance(image_caption_source_list, list):
        image_caption_source_list = [image_caption_source_list]

    res_folder = f"CONCH/results/{vl_model}"
    res_csv = os.path.join(res_folder, f"{vl_model}_{task}.csv")
    metrics = ["acc", "bacc", "weighted_kappa",
               "kappa", "roc_auc", "weighted_f1"]
    if not os.path.exists(res_csv):
        with open(res_csv, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "exp", "tr_imgs"]+metrics)
    parameter_csv = os.path.join(
        res_folder, f"{vl_model}_parameter_tuning.csv")
    if not os.path.exists(parameter_csv):
        with open(parameter_csv, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "name", "lr", "weight_decay"])

    modes = ["zero_shot_eval", "finetune", "linear_probing"]
    if isinstance(args.run_modes, int):
        args.run_modes = [args.run_modes]
    run_modes = [modes[i] for i in args.run_modes]

    for m in run_modes:
        if m == "zero_shot_eval":
            """
            Compose(
                Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
                CenterCrop(size=(224, 224))
                ToTensor()
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            )
            """
            model, preprocess, tokenizer = get_vl_model(vl_model)
            results = zero_shot_eval(
                vl_model, task, model, preprocess, tokenizer)
            line = [time.strftime("%Y-%m-%d %H:%M:%S"), "zero_shot", 0] + \
                [float(np.round(results[m], 4)) for m in metrics]
            with open(res_csv, mode="a") as file:
                writer = csv.writer(file)
                writer.writerow(line)

        if m == "finetune":
            for image_caption_source in image_caption_source_list:
                if image_caption_source == "ARCH":
                    image_dir = "data/ARCH"
                elif image_caption_source == "quilt1m":
                    image_dir = "data/Quilt_1M/quilt_1m"
                else:
                    raise ValueError

                training_images_list = args.training_images
                if not isinstance(training_images_list, list):
                    training_images_list = [training_images_list]
                retrieve_csv = f"CONCH/results/{task}_{image_caption_source}.csv"
                collected_pair_amout = len(pd.read_csv(retrieve_csv))
                training_images_list2 = [
                    ele for ele in training_images_list if ele < collected_pair_amout]
                training_images_list2 += [collected_pair_amout]
                for training_images in training_images_list2:
                    name = f"{m}_{task}_{image_caption_source}_tr_{training_images}"
                    print(
                        f"======================{name}=====================================")
                    df = pd.read_csv(parameter_csv)
                    row = df[df["name"] == name]
                    if len(row) > 0:
                        best_para = {"lr": row["lr"].item(),
                                     "weight_decay": row["weight_decay"].item()}
                    else:
                        best_para = finetune_parameter_tuning(vl_model=vl_model,
                                                              task=task,
                                                              image_caption_source=image_caption_source,
                                                              training_images=training_images,
                                                              image_dir=image_dir)
                        new_row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "name": name,
                                   "lr": best_para["lr"],
                                   "weight_decay": best_para["weight_decay"]}
                        df = pd.concat(
                            [df, pd.DataFrame([new_row])], ignore_index=True)
                        df.to_csv(parameter_csv, index=False)

                    for repeat in range(args.repeats):
                        eval_only = False
                        eval_ckp = None
                        exp = f"{name}_repeat_{repeat}"
                        print(
                            f"======================{exp}==============================")
                        df = pd.read_csv(res_csv)
                        row = df[df["exp"] == exp]
                        if len(row) > 0:
                            print("finished, skip")
                            continue
                        results = finetune(vl_model=vl_model,
                                           task=task,
                                           image_caption_source=image_caption_source,
                                           lr=best_para["lr"],
                                           wd=best_para["weight_decay"],
                                           max_epochs=50,
                                           training_images=training_images,
                                           image_dir=image_dir,
                                           eval_only=eval_only, eval_ckp=eval_ckp)
                        existing_exps = df[df["exp"].str.contains(
                            f"{m}_{task}_{image_caption_source}")]
                        best_metric = "weighted_kappa" if task == "SICAP" else "bacc"
                        if len(existing_exps) == 0:
                            best_performance = 0
                        else:
                            best_performance = max(
                                existing_exps[best_metric])

                        checkpoint_dir = f"logs/{vl_model}_finetune/{task}_{image_caption_source}/checkpoints/"
                        all_checkpoints = glob(
                            os.path.join(checkpoint_dir, "*.ckpt"))
                        latest_checkpoint = max(
                            all_checkpoints, key=os.path.getmtime)
                        if results[best_metric] > best_performance:
                            print(
                                f"new best performance, {results[best_metric]}")
                            shutil.copy(
                                src=latest_checkpoint, dst=f"log_models/{vl_model}_finetune/{task}_{image_caption_source}_best.ckpt")
                        os.remove(latest_checkpoint)
                        latest_event = glob(
                            f"logs/{vl_model}_finetune/{task}_{image_caption_source}/events.out*")[0]
                        os.remove(latest_event)
                        line = [time.strftime("%Y-%m-%d %H:%M:%S"), exp, training_images] + \
                            [float(np.round(results[m], 4)) for m in metrics]
                        with open(res_csv, mode="a") as file:
                            writer = csv.writer(file)
                            writer.writerow(line)

        if m == "linear_probing":
            five_fold_split_csv = f"CONCH/results/{task}_5_folds.csv"
            
            subset_size_list = args.subset_size
            if not isinstance(subset_size_list, list):
                subset_size_list = [subset_size_list]

            for image_caption_source in image_caption_source_list:
                if image_caption_source == "original":
                    finetune_ckp = None  # the original image encoder from the foundation model
                else:
                    finetune_ckp = f"log_models/{vl_model}_finetune/{task}_{image_caption_source}_best.ckpt"
                for subset_size in subset_size_list:
                    name = f"{m}_{task}_{image_caption_source}_subset_size_{subset_size}"
                    print(
                        f"======================{name}==================================")
                    df = pd.read_csv(parameter_csv)
                    row = df[df["name"] == name]
                    if len(row) > 0:
                        best_para = {"lr": row["lr"].item(),
                                     "weight_decay": row["weight_decay"].item()}
                    else:
                        best_para = linear_probing_parameter_tuning(vl_model=vl_model,
                                                                    task=task,
                                                                    five_fold_split_csv=five_fold_split_csv,
                                                                    finetune_ckp=finetune_ckp,
                                                                    subset_size=subset_size)
                        new_row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "name": name,
                                   "lr": best_para["lr"],
                                   "weight_decay": best_para["weight_decay"]}
                        df = pd.concat(
                            [df, pd.DataFrame([new_row])], ignore_index=True)
                        df.to_csv(parameter_csv, index=False)

                    for repeat in range(args.repeats):  # 5-fold corss-validation
                        eval_only = False
                        eval_ckp = None
                        exp = f"{name}_repeat_{repeat}"
                        print(
                            f"======================{exp}============================")
                        df = pd.read_csv(res_csv)
                        row = df[df["exp"] == exp]
                        if len(row) > 0:
                            print("finished, skip")
                            continue
                        training_images, results = linear_probing(vl_model=vl_model,
                                                 task=task,
                                                 five_fold_split_csv=five_fold_split_csv,
                                                 fold=repeat,
                                                 lr=best_para["lr"],
                                                 wd=best_para["weight_decay"],
                                                 max_epochs=50,
                                                 finetune_ckp=finetune_ckp,
                                                 subset_size=subset_size,
                                                 eval_only=eval_only, eval_ckp=eval_ckp)
                        
                        line = [time.strftime("%Y-%m-%d %H:%M:%S"), exp, training_images] + \
                            [float(np.round(results[m], 4)) for m in metrics]
                        with open(res_csv, mode="a") as file:
                            writer = csv.writer(file)
                            writer.writerow(line)


if __name__ == "__main__":
    main()
