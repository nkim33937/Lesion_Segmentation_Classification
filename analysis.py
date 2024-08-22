import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def calculate_relative_attenuation(image_array, mask_array):
    kidney_roi = mask_array == 1  # Assuming '1' is the label for kidney cortex
    lesion_roi = mask_array == 2  # Assuming '2' is the label for the lesion
    if np.any(kidney_roi) and np.any(lesion_roi):
        mean_kidney_hu = np.mean(image_array[kidney_roi])
        mean_lesion_hu = np.mean(image_array[lesion_roi])
        relative_attenuation = ((mean_lesion_hu - mean_kidney_hu) / mean_kidney_hu) * 100
        return relative_attenuation
    return None

def process_case(case_folder):
    image_path = os.path.join(case_folder, "instances", "imaging.nii.gz")
    mask_path = os.path.join(case_folder, "instances", "segmentation.nii.gz")
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    return calculate_relative_attenuation(image_array, mask_array)
                                                                    
def load_and_process_cases(base_dir):
    case_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('case_')]
    random.shuffle(case_folders)  # Shuffle to randomize dataset
    split_index = int(len(case_folders) * 0.8)
    train_folders = case_folders[:split_index]
    test_folders = case_folders[split_index:]
    train_data = [process_case(folder) for folder in train_folders]
    test_data = [process_case(folder) for folder in test_folders]
    return train_data, test_data

# Make sure to use the correct path
base_dir = "INSERT DATASET FOLDER PATH HERE"
train_data, test_data = load_and_process_cases(base_dir)

# Placeholder to visualize and analyze data
# You would normally include your analysis, plotting and threshold determination here
print(f"Training data relative attenuations: {train_data[:5]}")  # Output first few entries for verification
print(f"Test data relative attenuations: {test_data[:5]}")

# Placeholder for data labels (requires actual labels from your dataset)
train_labels = ['clear_cell_rcc' if random.random() > 0.5 else 'non_clear_cell_rcc' for _ in train_data]
test_labels = ['clear_cell_rcc' if random.random() > 0.5 else 'non_clear_cell_rcc' for _ in test_data]

# Analysis of training data
# Convert lists to DataFrame, assuming not None values are filtered
df_train = pd.DataFrame({
    'Relative Attenuation': [d for d in train_data if d is not None],
    'Label': train_labels
})

# Placeholder for actual data visualization and analysis, similar to previous plot and classification threshold determination examples

print("Processing complete. Implement plots and threshold determination based on your specific requirements.")