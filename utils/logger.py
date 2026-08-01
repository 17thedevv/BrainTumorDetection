import logging
import sys


def get_logger(name: str = "brain_tumor_detection", log_file: str = "training.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler - dung 'replace' de tranh UnicodeEncodeError tren Windows
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.stream.reconfigure(encoding='utf-8', errors='replace') if hasattr(ch.stream, 'reconfigure') else None

        # File handler - luon dung UTF-8
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
