import os

def check_labels_from_filenames(folder):
    labeled_images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            # Extract label from filename
            label = filename.split("(")[1].split(")")[0]
            label = label.zfill(3)  # Ensure label is represented as three-digit number
            image_path = os.path.join(folder, filename)
            print(f"Image: {filename}, Label: {label}")
            if label in labeled_images:
                labeled_images[label].append(image_path)
            else:
                labeled_images[label] = [image_path]
    return labeled_images

# Example usage:
folder = r"D:\MY WEBSITES\Drowsiness\face"
labeled_images = check_labels_from_filenames(folder)

# Print the grouped images for each label
for label, images in labeled_images.items():
    print(f"Label: {label}, Number of images: {len(images)}")
    for image in images:
        print(f"  {image}")

