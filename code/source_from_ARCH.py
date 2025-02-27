from glob import glob
import json
import numpy as np
import os
import pandas as pd
import string

from utils import semantic_sort_pair_with_CONCH
from downstream_dataset_preparation import *

def patterns(l):
    return [f" {l}, ", f" ({l})"]  # A, xxx. | (A) xxx. | xxx (A).


def which_pattern(splitter):
    if splitter.__contains__(","):
        return 0
    else:
        return 1


def get_letter(splitter):
    if which_pattern(splitter) == 0:
        return splitter[1]
    else:
        return splitter[2]


ALPHABETS = list(string.ascii_lowercase)
NUMBERS = [str(n) for n in range(1, 20)]
PATTERNS = sum([patterns(a) for a in ALPHABETS + NUMBERS], [])


def string_index(c, s):
    return [pos for pos, char in enumerate(s) if char == c]


def change_char_in_string(s, i, c):
    list_s = list(s)
    list_s[i] = c
    return ''.join(list_s)


def change_parentheses(s):
    # change (xxx) to [xxx], so that (x) can be detected as the subfigure splitter
    res = s
    left = string_index('(', res)[::-1]

    for i in left:
        first_right_loc = res[i + 1:].find(')')
        if first_right_loc > 1:
            res = change_char_in_string(res, i, '[')
            res = change_char_in_string(res, i + first_right_loc + 1, ']')
            left = string_index('(', res)[::-1]

    return res


def split(l, supcaption):
    supcaption = ' ' + supcaption
    splitters = [s for s in PATTERNS if s in supcaption]
    if len(splitters) <= 1:
        return supcaption
    # normalize the style of all splitters to the style of the first one, e.g., [' a, ', ' (b)'] --> [' a, ', ' b,']
    style = [which_pattern(s) for s in splitters]
    if any(np.array(style[1:]) - style[0]):
        new_splitters = [patterns(get_letter(s))[style[0]] for s in splitters]
        for i in range(1, len(splitters)):
            supcaption = supcaption.replace(splitters[i], new_splitters[i])
        splitters = new_splitters

    res = []
    if any([s + '.' in supcaption for s in splitters]):  # xxx (A).
        for s in splitters:
            if s not in supcaption:
                res.append('')
            else:
                # the subcaption is before the splitter
                res.append(supcaption.split(s)[0])
                supcaption = supcaption.split(s)[1]

        # assume the first sentence is the co-caption for all subfigures
        supcaption = ''.join(res[0].split('.')[:-1])
        res = [res[0], *[supcaption + '.' + r for r in res[1:]]]

    else:  # A, xxx. | (A) xxx.
        for s in splitters[::-1]:
            if s not in supcaption:
                res.append('')
            else:
                # the subcaption is after the splitter
                res.append(supcaption.split(s)[1])
                supcaption = supcaption.split(s)[0]
        res = [supcaption + r for r in res]
        res = res[::-1]

    try:
        return res[[l in s for s in splitters].index(True)]
    except:
        return supcaption


def get_classnames_text(task):
    exclude_list = ["and", "or", "the", "with",
                    "without", "in", "on"]
    exclude_list += ALPHABETS
    exclude_list += NUMBERS

    idx_to_class = get_idx_to_class(task)
    organ = get_task_organ(task)
    prompt_file = f'CONCH/prompts/{task}_prompts_all_per_class.json'
    with open(prompt_file) as f:
        prompts = json.load(f)['0']

    classnames = prompts['classnames']
    n_classes = len(classnames)
    classnames_text = sum([classnames[str(idx_to_class[idx])]
                          for idx in range(n_classes)], []) + list(classnames.keys())
    classnames_text = [t.replace(" tissue", "") for t in classnames_text]
    classnames_text = [t.split(",")[0] for t in classnames_text]
    classnames_text = [t.casefold() for t in classnames_text]
    classnames_text = sum([t.split(" ") for t in classnames_text], [])
    classnames_text = [t[:-1] if t[-1:] == "s" else t for t in classnames_text]
    classnames_text = [t[:-2] if t[-2:] == "es" else t for t in classnames_text]
    if any([t.__contains__("insitu") for t in classnames_text]):
        classnames_text += ["in situ", "in-situ"]
    classnames_text = set(classnames_text)
    classnames_text -= set(organ)
    classnames_text -= set(exclude_list)

    print(f"organ = {organ}")
    print(f"classnames_text = {classnames_text}")
    return organ, classnames_text


def collect_vl_data_from_ARCH(image_dir, caption_dir, retrieve_csv, task):
    organ, class_keys = get_keyword(task)
    with open(caption_dir, "r") as file:
        caption_dict = json.load(file)  # Load JSON into a dictionary
    retrieved = []
    for _, v in caption_dict.items():
        if any([k in v['caption'] for k in organ]) and any([k in v['caption'] for k in class_keys]):
            caption = v['caption'].casefold()
            caption = change_parentheses(caption)

            # split supcaptions
            if caption_dir.split("/")[-2] == "books_set" and v['letter'] != 'Single':
                caption = split(v['letter'].casefold(), caption)
            
            # it can happen that the caption doesn't contain key words after subcaption split
            if any([k in caption for k in organ]) and any([k in caption for k in class_keys]):
                v['caption'] = caption
                # print(caption)
                retrieved.append([image_dir+v["uuid"], caption])

    df = pd.DataFrame(retrieved, columns=['image_path', 'caption'])
    df.to_csv(retrieve_csv, index=False)


def get_suffix(image_dir, retrieve_csv):
    df = pd.read_csv(retrieve_csv)
    image_paths = df["image_path"].to_list()
    res = []
    for img_path in image_paths:
        suffix = glob(str(os.path.join(image_dir, img_path)) +
                      ".*")[0].split(".")[1]
        res.append(f"{img_path}.{suffix}")
    df['image_path'] = res
    df.to_csv(retrieve_csv, index=False)

def remove_duplicate(retrieve_csv):
    df = pd.read_csv(retrieve_csv)
    df.drop_duplicates(inplace=True)
    df.to_csv(retrieve_csv, index=False)

if __name__ == "__main__":
    task = "wsss4luad"
    # task = "BACH"
    # task = "SICAP"
    # task = "MHIST"
    retrieve_csv = f"CONCH/results/{task}_ARCH.csv"
    image_dir = "data/ARCH"

    for set_name in ["books_set", "pubmed_set"]:
        dataset_dir = f"{image_dir}/{set_name}"
        collect_vl_data_from_ARCH(image_dir=dataset_dir+"/images/", 
                                  caption_dir=os.path.join(dataset_dir, "captions.json"), 
                                  retrieve_csv=retrieve_csv.replace(".csv", f"_{set_name}.csv"),
                                  task=task)
    df_book = pd.read_csv(retrieve_csv.replace(".csv", f"_books_set.csv"))
    df_pubmed = pd.read_csv(retrieve_csv.replace(".csv", f"_pubmed_set.csv"))
    df = pd.concat([df_book, df_pubmed])
    df.to_csv(retrieve_csv, index=False)
    os.remove(retrieve_csv.replace(".csv", f"_books_set.csv"))
    os.remove(retrieve_csv.replace(".csv", f"_pubmed_set.csv"))
    get_suffix(image_dir, retrieve_csv)

    semantic_sort_pair_with_CONCH(retrieve_csv=retrieve_csv,
                                  output_csv=retrieve_csv,
                                  image_dir=image_dir)

