# python/nct_attack/logger.py
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# Глобальный конфиг логирования
_LOGGING_CONFIGURED = False
_LOG_LEVEL = logging.INFO
_LOG_DIR = Path("./logs")


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    file_logging: bool = True
) -> None:
    global _LOGGING_CONFIGURED, _LOG_LEVEL, _LOG_DIR
    
    if _LOGGING_CONFIGURED:
        return  
    
    _LOG_LEVEL = level
    if log_dir:
        _LOG_DIR = log_dir
    
    # директория для логов
    if file_logging:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # обработчик файла 
    if file_logging:
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / 'nct_attack.log',
            maxBytes=10_000_000,  
            backupCount=5
        )
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    
    if not _LOGGING_CONFIGURED:
        setup_logging()
    
    return logging.getLogger(name)
