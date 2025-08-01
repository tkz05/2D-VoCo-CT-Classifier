import sys
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
import pydicom

sys.path.append(os.path.abspath('H:/rsna-2023-abdominal-trauma-detection-main/src'))
import dicom_utilities


def write_image(dicom_file_path, output_directory, pixel_spacing=None):

    """
    Read DICOM file and write it as a png file

    Parameters
    ----------
    dicom_file_path: str
        Path of the DICOM file

    output_directory: pathlib.Path
        Path of the directory image will be written to

    pixel_spacing: tuple
        Image pixel spacing after normalization
    """

    dicom = pydicom.dcmread(str(dicom_file_path))
    image = dicom.pixel_array
    image = dicom_utilities.adjust_pixel_values(
        image=image, dicom=dicom,
        bits_allocated='dataset', bits_stored='dataset',
        rescale_slope='dataset', rescale_intercept='dataset',
        window_centers=['dataset'], window_widths=['dataset'],
        photometric_interpretation='dataset', max_pixel_value=1
    )

    if pixel_spacing is not None:
        image = dicom_utilities.adjust_pixel_spacing(
            image=image,
            dicom=dicom,
            current_pixel_spacing='dataset',
            new_pixel_spacing=pixel_spacing
        )
    output_file_path = os.path.join(output_directory, f'{os.path.splitext(os.path.basename(dicom_file_path))[0]}.png')
    cv2.imwrite(output_file_path, image)
    # cv2.imwrite(str(output_directory / f'{dicom_file_path.split("/")[-1].split(".")[0]}.png'), image)


if __name__ == '__main__':

    dicom_dataset_directory = "H:\\train_images"
    patient_ids = sorted(os.listdir(dicom_dataset_directory), key=lambda filename: int(filename))

    output_directory = "H:\\14th_output\\images"
    os.makedirs(output_directory, exist_ok=True)
    # output_directory.mkdir(parents=True, exist_ok=True)

    for patient in tqdm(patient_ids):

        # patient_directory = dicom_dataset_directory / patient
        patient_directory = os.path.join(dicom_dataset_directory, patient)
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan in patient_scans:
            # scan_directory = patient_directory / scan
            scan_directory = os.path.join(patient_directory, scan)
            file_names = sorted(os.listdir(scan_directory), key=lambda x: int(str(x).split('.')[0]))
               
            # scan_output_directory = output_directory / patient / scan
            # scan_output_directory.mkdir(parents=True, exist_ok=True)
            scan_output_directory = os.path.join(output_directory, patient, scan)
            os.makedirs(scan_output_directory, exist_ok=True)


            Parallel(n_jobs=8)(
                delayed(write_image)(
                    dicom_file_path=os.path.join(scan_directory, file_name),
                    output_directory=scan_output_directory,
                    # dicom_file_path=str(scan_directory / file_name),
                    # output_directory=scan_output_directory,
                    pixel_spacing=None
                )
                for file_name in file_names
            )

            # settings.logger.info(f'Finished writing {len(file_names)} images for patient {patient} scan {scan}')
