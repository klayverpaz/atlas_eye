from pathlib import Path
import numpy as np
import cv2

class FilesManager:

    @staticmethod
    def create_folder_if_not_exists(folder_path):
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True)
                print(f"Folder '{folder_path}' created successfully.")
            except Exception as e:
                print(f"Error creating folder '{folder_path}': {e}")
        else:
            print(f"Folder '{folder_path}' already exists.")

    @staticmethod
    def save_enhancedImages(enhanced_images:np.array, to_save_folder:str, images_prefix:str) -> list[str]:
        FilesManager.create_folder_if_not_exists(to_save_folder)
        images_paths = []
        for i, enhanced_image in enumerate(enhanced_images):
            new_image_name = images_prefix + str(i) + ".jpg"
            image_abs_path = to_save_folder + new_image_name
            cv2.imwrite(image_abs_path, enhanced_image)
            images_paths.append(image_abs_path)
            print(f"Image saved {image_abs_path}")
        return images_paths