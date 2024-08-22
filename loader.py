import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

def multiply_image_with_mask(image_path, mask_path, output_path):
    try:
        # Check if the input files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Read the input image and mask
        print(f"Reading image from {image_path}")
        image = sitk.ReadImage(image_path)
        print(f"Reading mask from {mask_path}")
        mask = sitk.ReadImage(mask_path)
        
        # Ensure the image and mask have the same size
        if image.GetSize() != mask.GetSize():
            raise ValueError("Image and mask must have the same dimensions.")
        
        # Convert the mask to the same pixel type as the image if necessary
        if image.GetPixelID() != mask.GetPixelID():
            mask = sitk.Cast(mask, image.GetPixelID())
        
        # Multiply the image by the mask
        result = sitk.Multiply(image, mask)
        
        # Save the result as a new NIfTI image
        print(f"Saving result to {output_path}")
        sitk.WriteImage(result, output_path)
    except Exception as e:
        print(f"Error in multiply_image_with_mask: {e}")

def find_brightest_voxel_and_draw_circle(image_path, mask_path, output_path):
    try:
        # Read the input image and mask
        print(f"Reading image from {image_path}")
        image = sitk.ReadImage(image_path)
        print(f"Reading mask from {mask_path}")
        mask = sitk.ReadImage(mask_path)

        # Convert to numpy arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        # Create a masked image where the lesion is isolated
        lesion_mask = (mask_array == 2)
        new_img = image_array * lesion_mask

        # Find the brightest voxel in the lesion area
        max_value = np.max(new_img)
        max_indices = np.argwhere(new_img == max_value)

        # Find the slice with the highest mean value within the lesion mask
        best_index = None
        best_mean_value = -np.inf

        for index in max_indices:
            slice_idx = index[0]
            circle_mask = np.zeros_like(new_img[slice_idx])
            y, x = np.ogrid[:circle_mask.shape[0], :circle_mask.shape[1]]
            distance = (y - index[1])**2 + (x - index[2])**2
            circle_mask[distance <= 30**2] = 1
            
            mean_value = np.mean(new_img[slice_idx][circle_mask == 1])
            
            if mean_value > best_mean_value:
                best_mean_value = mean_value
                best_index = index

        # Create a new mask with a circle around the brightest voxel on the best slice
        circle_mask = np.zeros_like(mask_array)
        slice_idx = best_index[0]
        y, x = np.ogrid[:circle_mask.shape[1], :circle_mask.shape[2]]
        distance = (y - best_index[1])**2 + (x - best_index[2])**2
        circle_mask[slice_idx][distance <= 30**2] = 1

        # Save the circle mask as a new NIfTI image
        circle_mask_img = sitk.GetImageFromArray(circle_mask)
        circle_mask_img.CopyInformation(image)
        print(f"Saving circle mask to {output_path}")
        sitk.WriteImage(circle_mask_img, output_path)

        return best_mean_value
    except Exception as e:
        print(f"Error in find_brightest_voxel_and_draw_circle: {e}")
        return None

if __name__ == "__main__":
    base_dir = "INSERT PATH HERE"

    # Load the JSON dataset once
    with open(os.path.join(base_dir, "INSERT JSON DATASET HERE"), 'r') as file:
        dataset = json.load(file)

    # Print the first few entries to understand the structure
    print(f"Dataset keys: {list(dataset[0].keys())}")
    print(f"First entry: {dataset[0]}")

    # Placeholder for updated dataset with mean HU values
    all_cases_data = []

    for subdir in os.listdir(base_dir):
        if not subdir.startswith("case_"):
            continue  # Skip non-case folders

        case_path = os.path.join(base_dir, subdir)

        # Define image, mask, and output paths within the case folder
        image_path = os.path.join(case_path, "imaging.nii.gz")
        mask_path = os.path.join(case_path, "segmentation.nii.gz")
        output_path = os.path.join(case_path, "outimg.nii.gz")

        # Executing functions
        try:
            mean_hu = find_brightest_voxel_and_draw_circle(image_path, mask_path, output_path)
            if mean_hu is not None:
                case_id = subdir  # Use the full case_id with "case_" prefix
                print(f"Processing case_id: {case_id}")

                # Find the corresponding entry in the dataset
                case_data = next((item for item in dataset if item['case_id'] == case_id), None)
                if case_data:
                    print(f"Found data for case_id {case_id}")
                    case_data['mean_hu'] = mean_hu
                    all_cases_data.append(case_data)
                else:
                    print(f"No data found for case_id {case_id}")
        except ValueError as e:
            print(f"Error processing case {subdir}: {e} \n")

    # Shuffle the collected data
    random.shuffle(all_cases_data)

    # Calculate split index and ensure it is an integer
    split_index = int(len(all_cases_data) * 0.8)

    # Split data into training and test sets
    train_data = all_cases_data[:split_index]
    test_data = all_cases_data[split_index:]

    # Save the test data to a separate file
    with open(os.path.join(base_dir, 'test_data.json'), 'w') as file:
        json.dump(test_data, file)

    # Debug print the collected data
    print(f"Collected train data: {len(train_data)} entries")
    print(f"Collected test data: {len(test_data)} entries")

    if not train_data:
        print("No training data collected.")
        exit()

    # Extract tumor subtypes and their counts
    subtype_counts = {}
    for entry in train_data:
        subtype = entry['tumor_histologic_subtype']
        if subtype not in subtype_counts:
            subtype_counts[subtype] = 0
        subtype_counts[subtype] += 1

    # Print subtype counts
    print("Subtype counts:")
    for subtype, count in subtype_counts.items():
        print(f"{subtype}: {count}")

    if not subtype_counts:
        print("No subtype counts available for plotting.")
        exit()

    # Filter out None values from subtypes
    subtype_counts = {k: v for k, v in subtype_counts.items() if k is not None}

    # Prepare data for plotting
    subtypes = []
    mean_hu = []
    for entry in train_data:
        if entry['tumor_histologic_subtype'] is not None:
            subtypes.append(entry['tumor_histologic_subtype'])
            mean_hu.append(entry['mean_hu'])

    # Debug print the data for plotting
    print(f"Subtypes: {subtypes}")
    print(f"Mean HU values: {mean_hu}")

    if not subtypes or not mean_hu:
        print("No data to plot.")
        exit()

    # Convert to DataFrame
    df = pd.DataFrame({'subtype': subtypes, 'mean_hu': mean_hu})

    # Bar Chart
    plt.figure(figsize=(10, 6))
    subtype_counts_sorted = dict(sorted(subtype_counts.items()))
    print(f"Plotting bar chart with subtypes: {list(subtype_counts_sorted.keys())}")
    plt.bar(subtype_counts_sorted.keys(), subtype_counts_sorted.values())
    plt.xlabel('Tumor Histologic Subtype')
    plt.ylabel('Count')
    plt.title('Counts of Tumor Histologic Subtypes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'tumor_subtype_counts_bar_chart.png'))

    # Save bar chart and close the plot to free memory
    plt.close()

    # Box Whisker Plot
    plt.figure(figsize=(10, 6))
    print(f"Plotting box whisker plot with subtypes: {df['subtype'].unique()}")
    df.boxplot(column='mean_hu', by='subtype', grid=False)
    plt.xlabel('Tumor Histologic Subtype')
    plt.ylabel('Mean Hounsfield Unit')
    plt.title('Mean Hounsfield Units by Tumor Subtype')
    plt.suptitle('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'tumor_subtype_hu_box_whisker.png'))

    # Save box whisker plot and close the plot to free memory
    plt.close()