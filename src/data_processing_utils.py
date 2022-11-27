import SimpleITK as sitk
import nibabel as nib
import numpy as np


def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def load_mask(file_patch):
    mask = nib.load(file_patch)
    mask = mask.get_fdata().transpose(2, 0, 1)
    return mask
