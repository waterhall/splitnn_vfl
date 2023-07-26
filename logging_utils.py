import logging
import sys

logging.getLogger("paramiko").setLevel(logging.WARNING)

root_logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout)