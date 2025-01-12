import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 创建 logs 目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 配置根日志记录器
logging.basicConfig(level=logging.INFO)

# 创建格式化器
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# 文件处理器
file_handler = RotatingFileHandler(
    log_dir / "workflow.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# 获取 logger
logger = logging.getLogger("workflow")
logger.setLevel(logging.INFO)

# 移除所有现有的处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 设置 propagate 为 False，防止日志重复
logger.propagate = False

# 测试日志是否正常工作
logger.info("Logger initialized successfully")
