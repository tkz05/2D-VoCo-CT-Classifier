import os
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
import pydicom
import sys
sys.path.append(os.path.abspath('H:/rsna-2023-abdominal-trauma-detection-main/src'))
import dicom_utilities


def write_image(dicom_file_path, output_directory):
    # print(f"Processing {dicom_file_path}")
    dicom = pydicom.dcmread(dicom_file_path)
    image = dicom.pixel_array
    image = dicom_utilities.adjust_pixel_values(
        image=image, dicom=dicom,
        bits_allocated='dataset', bits_stored='dataset',
        rescale_slope='dataset', rescale_intercept='dataset',
        window_centers=['dataset'], window_widths=['dataset'],
        photometric_interpretation='dataset', max_pixel_value=1
    )

    output_file_path = os.path.join(output_directory, f'{os.path.splitext(os.path.basename(dicom_file_path))[0]}.png')
    cv2.imwrite(output_file_path, image)
    # print(f"Saved image to {output_file_path}")


if __name__ == '__main__':
    print("Starting DICOM to PNG conversion...")
    dicom_dataset_directory = "H:\\train_images"
    patient_ids = [
        '19', '26575', '394', '851', '2232', '2429', '2986', '3194', '65504', '65360',
        '65149', '2384', '3414', '3785', '16731', '19384', '37022', '48508', '51038'
    ]

    # 動態讀取 train_images 資料夾中的所有病患資料夾
    patient_ids = sorted(os.listdir("H:/train_images"))
    
    output_directory = "H:\\14th_output\\images"
    os.makedirs(output_directory, exist_ok=True)

    for patient in tqdm(patient_ids):
        patient_directory = os.path.join(dicom_dataset_directory, patient)
        patient_scans = sorted(os.listdir(patient_directory), key=lambda filename: int(filename))

        for scan in patient_scans:
            scan_directory = os.path.join(patient_directory, scan)
            file_names = sorted(os.listdir(scan_directory), key=lambda filename: int(os.path.splitext(filename)[0]))

            scan_output_directory = os.path.join(output_directory, patient, scan)
            os.makedirs(scan_output_directory, exist_ok=True)

            Parallel(n_jobs=8)(
                delayed(write_image)(
                    dicom_file_path=os.path.join(scan_directory, file_name),
                    output_directory=scan_output_directory
                )
                for file_name in tqdm(file_names)
            )

            # print(f'Finished writing {len(file_names)} images for patient {patient} scan {scan}')
