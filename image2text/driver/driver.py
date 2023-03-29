import logging
import signal
import sys
from PIL import Image
from multiprocessing import Pool, cpu_count
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.Logger('status', logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def handle_sigint(*args):
    exit(0)

def init_model_worker():
    """Инициализирует подпроцесс с моделью

    Каждый подпроцесс содержит отдельный экземпляр модели
    который ожидает изображение для обработки и возвращает 
    описание картинки
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        global model, processor, worker_logger
        worker_logger = logging.Logger("worker", level=logging.INFO)
        worker_handler = logging.StreamHandler(sys.stdout)
        worker_handler.setFormatter(logging.Formatter("%(levelname)s:\t %(asctime)s -- [%(process)d] -- \t%(message)s"))
        worker_logger.addHandler(worker_handler)

        worker_logger.info("Worker is initializing...")

        # Может быть использована другая image2text модель
        # с HuggingFace
        processor = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps")

        worker_logger.info("Worker is initialized.")
        
        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)
    except:
        exit(0)

def process_image(img_path: str):
    """Обрабатывает изображение
    
    Создаёт описание к полученному на вход изображению
    используя image2text модель инициализированную в
    подпроцессе

    Args:
        img_path (str): Путь к изображению

    Returns:
        str: Полученное описание
    """
    global model, processor, worker_logger
    model.eval()
    try:
        worker_logger.info(f"Reading... {img_path[-20:]}")
        worker_logger.info(f"Processing... {img_path[-20:]}")

        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(pixel_values=inputs.pixel_values, max_length=20)
        out = processor.batch_decode(out, skip_special_tokens=True)[0]
        
        worker_logger.info(f"Processed {img_path[-20:]}.")
        return out
    except KeyboardInterrupt:
        exit(0)


class Engine:
    def __init__(self, workers=cpu_count() // 2):
        self.pool = Pool(workers, initializer=init_model_worker)
        
    def process_images(self, image_paths):
        try:
            results = []
            for image_path in image_paths:
                results.append(self.pool.apply_async(process_image, args=(image_path,)))
            return [x.get() for x in results]
        except KeyboardInterrupt:
                self.pool.close()
                self.pool.terminate()
                self.pool.join()



