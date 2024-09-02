import logging


class ColorHandler(logging.StreamHandler):
    def format(self, record):
        # Определяем ANSI-коды для цветов
        GRAY = "\033[90m"
        WHITE = "\033[97m"
        RED = "\033[91m"
        RESET = "\033[0m"
        # Форматируем запись
        formatted_record = super().format(record)
        # Проверяем уровень логирования и задаем цвет для всей строки
        if record.levelno == logging.DEBUG:
            return f"{GRAY}{formatted_record}{RESET}"
        elif record.levelno == logging.INFO:
            return f"{WHITE}{formatted_record}{RESET}"
        else:
            return f"{RED}{formatted_record}{RESET}"


class LOGGER:
    def __init__(self, level: str = 'DEBUG'):
        # Создаем логгер
        self.logger = logging.getLogger("colored_logger")
        if level.lower() == 'debug':
            self.logger.setLevel(logging.DEBUG)
            # Форматируем сообщение с учетом времени, пути файла и строки, включая трассировку ошибок
            formatter = logging.Formatter('%(levelname)-9s| %(asctime)s | %(lineno)d - %(pathname)s | %(message)s',
                                          datefmt='%d-%m-%Y %H:%M:%S')
        elif level.lower() == 'info':
            self.logger.setLevel(logging.INFO)
            # Форматируем сообщение с учетом времени, пути файла и строки, включая трассировку ошибок
            formatter = logging.Formatter('%(levelname)-9s| %(message)s')
        elif level.lower() == 'warning':
            self.logger.setLevel(logging.WARNING)
            # Форматируем сообщение с учетом времени, пути файла и строки, включая трассировку ошибок
            formatter = logging.Formatter('%(levelname)-9s| %(asctime)s | %(lineno)d - %(pathname)s | %(message)s',
                                          datefmt='%d-%m-%Y %H:%M:%S')
        elif level.lower() == 'error':
            self.logger.setLevel(logging.ERROR)
            # Форматируем сообщение с учетом времени, пути файла и строки, включая трассировку ошибок
            formatter = logging.Formatter('%(levelname)-9s| %(asctime)s | %(lineno)d - %(pathname)s | %(message)s',
                                          datefmt='%d-%m-%Y %H:%M:%S')
        elif level.lower() == 'critical':
            self.logger.setLevel(logging.CRITICAL)
            # Форматируем сообщение с учетом времени, пути файла и строки, включая трассировку ошибок
            formatter = logging.Formatter('%(levelname)-9s| %(asctime)s | %(lineno)d - %(pathname)s | %(message)s',
                                          datefmt='%d-%m-%Y %H:%M:%S')
        else:
            raise 'Неизвестный уровень логирования'

        # Настраиваем кастомный обработчик
        color_handler = ColorHandler()

        color_handler.setFormatter(formatter)

        self.logger.addHandler(color_handler)

    def get_logger(self):
        return self.logger
