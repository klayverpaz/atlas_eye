from skimage.metrics import structural_similarity
import numpy as np
import pandas as pd

class Metrics:

    @staticmethod
    def RMSE(imagem1, imagem2):
        if imagem1.shape != imagem2.shape:
            raise ValueError("As imagens têm dimensões diferentes.")
        erro_quadratico = np.square(imagem1.astype(np.float32) - imagem2.astype(np.float32))
        rmse = np.sqrt(np.mean(erro_quadratico))
        return rmse
    @staticmethod
    def AMBE(imagem1, imagem2):
        if imagem1.shape != imagem2.shape:
            raise ValueError("As imagens têm dimensões diferentes.")
        diferenca_absoluta = np.abs(imagem1.astype(np.float32) - imagem2.astype(np.float32))
        ambe = np.mean(diferenca_absoluta)
        return ambe
    
    @staticmethod
    def PSNR(imagem1, imagem2):
        if imagem1.shape != imagem2.shape:
            raise ValueError("As imagens têm dimensões diferentes.")
        mse = np.mean((imagem1 - imagem2) ** 2)
        max_pixel = np.max(imagem1)
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr
    
    @staticmethod
    def SSIM(imagem1, imagem2):
        return structural_similarity(imagem1, imagem2, data_range=imagem2.max() - imagem2.min())
    

    @staticmethod
    def get_metrics(imagem1, imagem2):
        metrics_dict = {}
        metrics_dict["RMSE"] = Metrics.RMSE(imagem1, imagem2)
        metrics_dict["AMBE"] = Metrics.AMBE(imagem1, imagem2)
        metrics_dict["PSNR"] = Metrics.PSNR(imagem1, imagem2)
        metrics_dict["SSIM"] = Metrics.SSIM(imagem1, imagem2)
        return metrics_dict

    @staticmethod
    def average_metrics(image_tuples):
        """_summary_

        Args:
            image_tuples (list[(np:array, np:array)]): 

        Returns:
            _type_: dict
        """
        num_images = len(image_tuples)
        if num_images == 0:
            return None
        
        total_metrics = {
            "RMSE": 0,
            "AMBE": 0,
            "PSNR": 0,
            "SSIM": 0
        }

        for original_img, enhanced_img in image_tuples:
            metrics = Metrics.get_metrics(original_img, enhanced_img)
            for key, value in metrics.items():
                total_metrics[key] += value

        average_metrics = {key: value / num_images for key, value in total_metrics.items()}
        return average_metrics