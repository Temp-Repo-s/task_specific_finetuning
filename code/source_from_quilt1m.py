import json
import pandas as pd

from source_from_ARCH import get_keyword
from utils import semantic_sort_pair_with_CONCH


def data_clean():
    df = pd.read_csv(f"{quilt_root}/quilt_1M_lookup.csv")
    """
    ['Unnamed: 0', 'caption', 'image_path', 'subset', 'split', 'pathology',
        'roi_text', 'noisy_text', 'corrected_text', 'med_umls_ids',
        'magnification', 'height', 'width']
    """
    # step 1: make the lookup table smaller by keeping only image_path and caption
    columns_to_keep = ["image_path", "caption"]
    new_df = df[columns_to_keep]
    # length: 1017712
    new_df.to_csv(f"{quilt_root}/quilt_1M_lookup_filename_caption_only.csv")

    # step 2: make the lookup table smaller by keeping only clean images
    df_cleaner = pd.read_csv(f"{quilt_root}/predictions_quiltcleaner.csv")
    """
    ['Unnamed: 0', 'Persons/Photos', 'Desktop/Windows/SlideViewer', 'Text/Logo in Image', 'Arrow/Annotation', 'Image Perspective/Quality', 'Additional (On-Slide) Overview', 'Additional Control Elements', 'Multi-Panel Image',
        'Any impairment', 'Filename']
    """
    df_cleaner_train = pd.read_csv(f"{quilt_root}/train_annotations.csv")
    df_cleaner_val = pd.read_csv(f"{quilt_root}/val_annotations.csv")
    df_cleaner_test = pd.read_csv(f"{quilt_root}/test_annotations.csv")
    """
    ['Image', 'Persons/Photos', 'Desktop/Windows/SlideViewer', 'Text/Logo in Image', 'Arrow/Annotation', 'Image Perspective/Quality', 'Additional (On-Slide) Overview', 'Additional Control Elements', 'Multi-Panel Image']
    """

    clean_images = []
    for index, row in df_cleaner.iterrows():
        if row["Any impairment"] < 0.5:
            clean_images.append(row["Filename"])

    for df in [df_cleaner_train, df_cleaner_val, df_cleaner_test]:
        for index, row in df.iterrows():
            if not any(row[['Persons/Photos', 'Desktop/Windows/SlideViewer', 'Text/Logo in Image', 'Arrow/Annotation', 'Image Perspective/Quality', 'Additional (On-Slide) Overview', 'Additional Control Elements', 'Multi-Panel Image']].to_numpy()):
                clean_images.append(row["Image"].replace("images/", ""))
    print(f"len(clean_images): {len(clean_images)}")
    with open(f"{quilt_root}/clean_images.json", 'w') as f:
        json.dump(clean_images, f)

    csv_file = f"{quilt_root}/quilt_1M_lookup_filename_caption_only.csv"
    df = pd.read_csv(csv_file)
    filtered_rows = df[df["image_path"].isin(clean_images)]
    print(len(filtered_rows))
    pd.DataFrame(filtered_rows).to_csv(
        f"{quilt_root}/quilt_1M_lookup_filename_caption_clean.csv", index=False)  # lengeh: 232039


def collect_vl_data_from_quilt1m(caption_dir, retrieve_csv, task):
    organ, class_keys = get_keyword(task)
    df = pd.read_csv(caption_dir)
    retrieved = []
    for index, row in df.iterrows():
        if any([k in row['caption'] for k in organ]) and any([k in row['caption'] for k in class_keys]):
            retrieved.append(row)

    # print(len(retrieved))
    pd.DataFrame(retrieved).to_csv(
        retrieve_csv, index=False)  # wsss4luad : 649


def check(retrieve_csv):
    # check if the final csv file aligns with the given official one, just in case there is any mistake in csv and pd dataframe operations
    df = pd.read_csv(f"{quilt_root}/quilt_1M_lookup_filename_caption_only.csv")
    retrieve_df = pd.read_csv(retrieve_csv)
    
    merged = retrieve_df.merge(df, how='left', indicator=True)
    missing_rows = merged[merged['_merge'] ==
                          'left_only'].drop('_merge', axis=1)
    still_missing_rows = []
    for idx, row in missing_rows.iterrows():
        id = row["Unnamed: 0"]
        for col in ["image_path", "caption"]:
            orig_value = df[df["Unnamed: 0"] == id][col].item()
            if orig_value.replace("\n", " ") != row[col]:
                still_missing_rows.append([id, col])
                print(id, f"{col} wrong")
                print(row[col])
                print(orig_value)
    assert len(still_missing_rows) == 0


quilt_root = "data/Quilt_1M"
if __name__ == "__main__":
    image_dir = f"{quilt_root}/quilt_1m"
    caption_dir = f"{quilt_root}/quilt_1M_lookup_filename_caption_clean.csv"
    task = "wsss4luad"
    # task = "BACH"
    # task = "CRC"
    # task = "SICAP"
    # task = "MHIST"

    retrieve_csv = f"CONCH/results/{task}_quilt1m.csv"
    collect_vl_data_from_quilt1m(caption_dir, retrieve_csv, task)
    # check(retrieve_csv)
    semantic_sort_pair_with_CONCH(retrieve_csv, output_csv=retrieve_csv, image_dir=image_dir)