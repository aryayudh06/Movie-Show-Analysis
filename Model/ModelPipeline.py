import subprocess
import logging
import os
from typing import Optional, Dict, Any

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # direktori Model/
        self.dataset_importer_script = os.path.join(self.base_dir, "DatasetImporter.py")
        self.model_script = os.path.join(self.base_dir, "model.py")
        self.model_predict_script = os.path.join(self.base_dir, "model_predict.py")
        self.model_dir = os.path.join(self.base_dir, "movie_genre_classifier")

    def model_exists(self) -> bool:
        """Check if the trained model exists"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "label_encoder_classes.npy",
            "label_encoder_params.npy"
        ]
        
        if not os.path.exists(self.model_dir):
            return False
            
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                return False
        return True

    def run_script(self, script_path: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """Menjalankan script Python dengan subprocess"""
        if not os.path.exists(script_path):
            logger.error(f"File tidak ditemukan: {script_path}")
            return False

        try:
            # Lokasi Python interpreter di dalam venv (Windows)
            venv_python = os.path.join(self.base_dir, "..", "venv", "Scripts", "python.exe")
            venv_python = os.path.abspath(venv_python)

            if not os.path.exists(venv_python):
                logger.error(f"Python di venv tidak ditemukan: {venv_python}")
                return False

            cmd = [venv_python, script_path]
            if args:
                for key, value in args.items():
                    cmd.extend([f"--{key}", str(value)])
            
            logger.info(f"Menjalankan script: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if result.stdout:
                logger.info(f"Output dari {os.path.basename(script_path)}:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Error dari {os.path.basename(script_path)}:\n{result.stderr}")
                
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Gagal menjalankan {os.path.basename(script_path)}: {str(e)}")
            logger.error(f"Error output:\n{e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error tak terduga saat menjalankan {os.path.basename(script_path)}: {str(e)}")
            return False
    
    def run_pipeline(self) -> bool:
        """Menjalankan seluruh pipeline"""
        logger.info("Memulai pipeline...")
        
        # Check if model exists
        if self.model_exists():
            logger.info("Model sudah ada, menjalankan prediksi...")
            return self.run_script(self.model_predict_script)
        else:
            # Langkah 1: Jalankan DatasetImporter untuk mendapatkan data
            logger.info("Model belum ada, menjalankan pipeline training...")
            logger.info("Langkah 1: Menjalankan DatasetImporter...")
            success = self.run_script(self.dataset_importer_script)
            if not success:
                logger.error("Gagal menjalankan DatasetImporter, pipeline dihentikan")
                return False
            
            # Langkah 2: Jalankan model training
            logger.info("Langkah 2: Menjalankan model training...")
            success = self.run_script(self.model_script)
            if not success:
                logger.error("Gagal menjalankan model training")
                return False
            
            logger.info("Pipeline training selesai dijalankan dengan sukses")
            return True

if __name__ == "__main__":
    pipeline = PipelineRunner()
    pipeline.run_pipeline()