from conch.open_clip_custom import create_model_from_pretrained
import os
import pandas as pd
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from conch.open_clip_custom import get_tokenizer, tokenize
import open_clip


read_token = "xxx"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_model(model_cfg='conch_ViT-B-16'):

    force_image_size = 224
    model, preprocess = create_model_from_pretrained(model_cfg,
                                                     checkpoint_path="hf_hub:MahmoodLab/conch",
                                                     hf_auth_token=read_token,
                                                     device=device,
                                                     force_image_size=force_image_size)
    return model, preprocess

def get_conch():
    model, preprocess = get_model()
    tokenizer = get_tokenizer()
    return model, preprocess, tokenizer


def get_quiltnet():
    model_name = 'hf-hub:wisdomik/QuiltNet-B-32'
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess_val, tokenizer


def get_vl_model(vl_model, ckp=None):
    if vl_model == "conch":
        model, preprocess, tokenizer = get_conch()
    elif vl_model == "quiltNet":
        model, preprocess, tokenizer = get_quiltnet()
    else:
        raise ValueError
    
    if ckp is not None:
        print(f"loading ckp {ckp}")
        checkpoint = torch.load(ckp)
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict.update({k.replace("model.", ""): v})
        model.load_state_dict(state_dict)

    return model, preprocess, tokenizer


def load_linear_probing_model_from_ckp(linear_probing_model, ckp):

    checkpoint = torch.load(ckp)
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.__contains__("encoder"):
            state_dict.update({k.replace("encoder.", ""): v})
    linear_probing_model.encoder.load_state_dict(state_dict)

    state_dict = {"weight": checkpoint["state_dict"]["classifier.weight"],
                  "bias": checkpoint["state_dict"]["classifier.bias"]}
    linear_probing_model.classifier.load_state_dict(state_dict)

    return linear_probing_model


def load_distill_from_ckp(model, ckp):

    checkpoint = torch.load(ckp)
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.__contains__("student"):
            state_dict.update({k.replace("student.", ""): v})

    model.load_state_dict(state_dict)
    return model


def get_train_val_split(five_fold_split_csv, folds, subset_size=-1, val_ratio=None, random_seed=42):
    if not isinstance(folds, list):
        folds = [folds]
    df = pd.read_csv(five_fold_split_csv)
    df = df[df["Fold"].isin(folds)]
    image_paths = df["Filename"].tolist()
    labels = df["Label"].tolist()

    if subset_size == -1:
        subset_size = len(image_paths)
    else:
        subset_size = min(len(image_paths), subset_size)
        print(f"randomly selecting {subset_size} data...")
        ds = list(zip(image_paths, labels))
        random.seed(random_seed)
        subset = random.sample(ds, subset_size)
        image_paths, labels = zip(*subset)
    print(f"{len(image_paths)} images selected")
    print(image_paths[:10])

    if val_ratio is not None:
        val_size = int(subset_size * val_ratio)
        val_image_paths = image_paths[:val_size]
        val_labels = labels[:val_size]
        train_image_paths = image_paths[val_size:]
        train_labels = labels[val_size:]
        print(f"{len(train_image_paths)} training images, {len(val_image_paths)} validation images")
        print(train_image_paths[:5])
        print(train_labels[:5])
        print(val_image_paths[:5])
        print(val_labels[:5])
        return train_image_paths, train_labels, val_image_paths, val_labels
    else:
        return image_paths, labels



class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB") 
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class ImageCaptionDataset(Dataset):
    def __init__(self, vl_model, retrieve_csv, transform, tokenizer=None, training_images=None, image_dir=None):
        df = pd.read_csv(retrieve_csv)
        self.images = df["image_path"].tolist()
        self.captions = df["caption"].tolist()

        if training_images is not None:
            self.images = self.images[:training_images]
            self.captions = self.captions[:training_images]

        check_image_path = self.images[0].__contains__("/data_rd")  # no dataset root
        if not check_image_path:
            self.images = [os.path.join(image_dir, img_path)
                           for img_path in self.images]

        self.transforms = transform
        self.tokenizer = tokenizer
        self.vl_model = vl_model
        print(f"{len(self.images)} image-caption pairs loaded")
        print(self.captions[0])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.vl_model == "conch":
            text_tokens = tokenize(texts=[str(self.captions[idx])], tokenizer=self.tokenizer).squeeze(0)
        elif self.vl_model == "quiltNet":
            text_tokens = self.tokenizer(str(self.captions[idx]))[0]
        else:
            raise ValueError
        return images, text_tokens


class ImageCaptionNoTokenizationDataset(Dataset):
    def __init__(self, retrieve_csv, transform, training_images=None, image_dir=None):
        df = pd.read_csv(retrieve_csv)
        self.images = df["image_path"].tolist()
        self.captions = df["caption"].tolist()

        if training_images is not None:
            self.images = self.images[:training_images]
            self.captions = self.captions[:training_images]

        check_image_path = self.images[0].__contains__("/data_rd")  # no dataset root
        if not check_image_path:
            self.images = [os.path.join(image_dir, img_path)
                           for img_path in self.images]

        self.transforms = transform
        print(
            f"ImageCaptionDataset: {len(self.images)} image-caption pairs loaded")
        # print(self.images)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts=[str(self.captions[idx])]
        return images, texts


def semantic_sort_pair_with_CONCH(retrieve_csv, output_csv, image_dir=None):
    model, preprocess = get_model()
    tokenizer = get_tokenizer()
    ds = ImageCaptionDataset(vl_model="conch",
                            retrieve_csv=retrieve_csv, 
                             tokenizer=tokenizer,
                             transform=preprocess,
                             image_dir=image_dir)
    dl = DataLoader(ds, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=True, drop_last=False)

    _ = model.eval()
    scores = []
    for batch_idx, batch in enumerate(dl):
        images, captions = batch
        image_embeddings = model.encode_image(images.to(device))
        text_embeddings = model.encode_text(captions.to(device))
        scores.append(torch.diag(model.logit_scale *
                      image_embeddings @ text_embeddings.T).clone().detach())

    scores = torch.concat(scores).cpu().numpy()
    df = pd.read_csv(retrieve_csv)
    df["align_score_CONCH"] = scores
    df_sorted = df.sort_values('align_score_CONCH', ascending=False)
    df_sorted.to_csv(output_csv, index=False)

    # fig, axes = plt.subplots(2, 4)
    # axes = axes.ravel()
    # for idx, ax in enumerate(axes):
    #     img_path = os.path.join(image_dir, df_sorted[idx:idx+1]["image_path"].item())
    #     img = Image.open(str(img_path))
    #     caption = df_sorted[idx:idx+1]["caption"].item()
    #     ax.imshow(img)
    #     ax.set_title(caption)
    # fig.savefig("CONCH/results/tmp.png")
    # plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
