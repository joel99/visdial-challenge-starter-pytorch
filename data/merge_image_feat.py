import argparse

import h5py

# Original use case is to combine downloaded image features (COCO train, extracted visdial val/test)
parser = argparse.ArgumentParser(
    description="Merge h5 files containing features of different splits of a dataset")

parser.add_argument(
    "--save-path",
    help="Path for merged output file",
    default="data_img_merged.h5"
)

parser.add_argument(
    "--train-path",
    help="Path for train file",
    default="data/data_img_train.h5"
)

parser.add_argument(
    "--val-path",
    help="Path for val file",
    default="data/data_img_val.h5"
)

parser.add_argument(
    "--test-path",
    help="Path for test file",
    default="data/data_img_test.h5"
)

def merge(input_files, output_path):
    """Given input files containing features over different splits of dataset, merge using external links.
    
    Parameters
    ----------
    input_files : str
        Dictionary keyed by split, containing input file paths
    output_path : str
        Path to save output as
    """
    
    # Warning - external links fail if referenced files are open
    # Assumed constant name for relevant features in input_files
    features_dataset_name = "/features"
    # TODO: modify tsv conversion to match this expectation
    with h5py.File(output_path) as merge_file:
        for split in input_files:
            merge_file[split] = h5py.ExternalLink(input_files[split], features_dataset_name)

def main(args):
    input_files = {'train': args.train_path,
                   'val': args.val_path,
                   'test': args.test_path}
    merge(input_files, args.save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)