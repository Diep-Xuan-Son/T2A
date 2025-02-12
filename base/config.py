import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

# from base.libs import *
import os
import time
import json
import shutil 
import logging 
import datetime
import requests
import re as regex
from enum import IntEnum
from loguru import logger
from typing import Optional
from pydantic import BaseModel

def check_folder_exist(*args, **kwargs):
	if len(args) != 0:
		for path in args:
			if not os.path.exists(path):
				os.makedirs(path, exist_ok=True)

	if len(kwargs) != 0:
		for path in kwargs.values():
			if not os.path.exists(path):
				os.makedirs(path, exist_ok=True)

def delete_folder_exist(*args, **kwargs):
	if len(args) != 0:
		for path in args:
			if os.path.exists(path):
				if os.path.isfile(path):
					os.remove(path)
				elif os.path.isdir(path):
					shutil.rmtree(path)

	if len(kwargs) != 0:
		for path in kwargs.values():
			if os.path.exists(path):
				if os.path.isfile(path):
					os.remove(path)
				elif os.path.isdir(path):
					shutil.rmtree(path)

class PathDefault(BaseModel):
	LOGDIR: Optional[str] = f"{str(ROOT)}/logs"
	
	def check_exist(self):
		check_folder_exist(**self.__dict__)
		print("----Check finished!")
PATH_DEFAULT = PathDefault()
PATH_DEFAULT.check_exist()

#---------------------------log---------------------------
logger.level("INFO", color="<light-green><dim>")
logger.level("DEBUG", color="<cyan><bold><italic>")
logger.level("WARNING", color="<yellow><bold><italic>")
logger.level("ERROR", color="<red><bold>")

# logger_app = build_logger("app_server", "app_server.log")
# logger_controller = build_logger("controller", "controller.log")
# logger_retrieval = build_logger("retrieval_worker", "retrieval_worker.log")

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.terminal = sys.stdout
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def __getattr__(self, attr):
		return getattr(self.terminal, attr)

	def write(self, buf):
		temp_linebuf = self.linebuf + buf
		self.linebuf = ''
		for line in temp_linebuf.splitlines(True):
			# From the io.TextIOWrapper docs:
			#   On output, if newline is None, any '\n' characters written
			#   are translated to the system default line separator.
			# By default sys.stdout.write() expects '\n' newlines and then
			# translates them so this is still cross platform.
			if line[-1] == '\n':
				self.logger.log(self.log_level, line.rstrip())
			else:
				self.linebuf += line

	def flush(self):
		if self.linebuf != '':
			self.logger.log(self.log_level, self.linebuf.rstrip())
		self.linebuf = ''

formatter = logging.Formatter(
	# fmt='\033[0;32m'+"%(asctime)s.%(msecs)03d | %(levelname)s    | %(name)s | %(message)s",
	fmt="%(asctime)s.%(msecs)03d | %(levelname)s    | %(name)s | %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S"
)

if not logging.getLogger().handlers:
	logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(formatter)
    
stdout_logger = logging.getLogger("stdout")
stdout_logger.setLevel(logging.INFO)
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger("stderr")
stderr_logger.setLevel(logging.ERROR)
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl
#////////////////////////////////////////////////////////////

def set_log_file(file_name="logger_app"):
	logger_app = logger.bind(name=file_name)
	logger_app.add(os.path.join(PATH_DEFAULT.LOGDIR, f"{file_name}.{datetime.date.today()}.log"), mode='w')
	return logger_app

def read_jsonline(address):
	not_mark = []
	with open(address, 'r', encoding="utf-8") as f:
		for jsonstr in f.readlines():
			jsonstr = json.loads(jsonstr)
			not_mark.append(jsonstr)
	return not_mark

def read_json(address):
	with open(address, 'r', encoding='utf-8') as json_file:
		json_data = json.load(json_file)
	return json_data

def write_json(address, data):
	with open(address, 'w') as json_file:
		json.dump(data, json_file, indent=4)

def estimate_execute_time(name_func: str, logger_: object):
	def decorator(func):
		def inner(*args, **kwargs):
			logger_.debug(name_func)
			st_time = time.time()
			value = func(*args, **kwargs)
			logger_.info("----Duration: {}", time.time()-st_time)
			return value
		return inner
	return decorator
