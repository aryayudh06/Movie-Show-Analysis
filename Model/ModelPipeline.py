import subprocess
import logging
from typing import Optional, Dict, Any

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.dataset_importer_script = "DatasetImporter.py"
        self.model_script = "model.py"
        
    def run_script(self, script_path: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """Menjalankan script Python dengan subprocess"""
        try:
            cmd = ["python", script_path]
            if args:
                for key, value in args.items():
                    cmd.extend([f"--{key}", str(value)])
            
            logger.info(f"Menjalankan script: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                logger.info(f"Output dari {script_path}:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Error dari {script_path}:\n{result.stderr}")
                
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Gagal menjalankan {script_path}: {str(e)}")
            logger.error(f"Error output:\n{e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error tak terduga saat menjalankan {script_path}: {str(e)}")
            return False
    
    def run_pipeline(self) -> bool:
        """Menjalankan seluruh pipeline"""
        logger.info("Memulai pipeline...")
        
        # Langkah 1: Jalankan DatasetImporter untuk mendapatkan data
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
        
        logger.info("Pipeline selesai dijalankan dengan sukses")
        return True

if __name__ == "__main__":
    pipeline = PipelineRunner()
    pipeline.run_pipeline()