import click
import os
from sklearn.model_selection import train_test_split
import glob
from pathlib import Path
from data_processing_utils import load_dicom, load_mask


@click.command()
@click.argument('feat_directory', type=click.Path(exists=True))
@click.argument('mask_directory', type=click.Path(exists=True))
@click.argument('output_directory_train', type=click.Path())
@click.argument('output_directory_test', type=click.Path())
@click.argument('seed', type=click.INT, default=101)
def train_val_split(feat_directory, mask_directory, output_directory_train, output_directory_test, seed):
    train, val = train_test_split(list(os.scandir(feat_directory)), random_state=seed)
    for lung in train:
        path = Path(glob.glob(os.path.join(lung.path, "*/*/*[!json]"))[0]).absolute()
        feat = load_dicom(r'{}'.format(path))
        path = Path(glob.glob(os.path.join(mask_directory, lung.name) + '/*')[0]).absolute()
        mask = load_mask(path)
        feat_path = os.path.join(output_directory_train, 'feat')
        mask_path = os.path.join(output_directory_train, 'mask')
        try:
            os.mkdir(output_directory_train)
            os.mkdir(feat_path)
            os.mkdir(mask_path)
        except FileExistsError:
            pass
        feat.dump(os.path.join(feat_path, lung.name))
        mask.dump(os.path.join(mask_path, lung.name))

    for lung in val:
        path = Path(glob.glob(os.path.join(lung.path, "*/*/*[!json]"))[0]).absolute()
        feat = load_dicom(r'{}'.format(path))
        path = Path(glob.glob(os.path.join(mask_directory, lung.name) + '/*')[0]).absolute()
        mask = load_mask(path)
        feat_path = os.path.join(output_directory_test, 'feat')
        mask_path = os.path.join(output_directory_test, 'mask')
        try:
            os.mkdir(output_directory_test)
            os.mkdir(feat_path)
            os.mkdir(mask_path)
        except FileExistsError:
            pass
        feat.dump(os.path.join(feat_path, lung.name))
        mask.dump(os.path.join(mask_path, lung.name))


if __name__ == '__main__':
    train_val_split()
