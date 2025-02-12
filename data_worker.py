import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
DIR = FILE.parents[0]
if DIR not in sys.path:
	sys.path.append(str(DIR))

import os
import re
import time
import queue
import threading
import traceback
import unicodedata
import multiprocessing as mp
from typing import Iterable, List, Optional, Union

from nat.config import FLAGS
from base.config import logger, set_log_file

class DataWorker(threading.Thread):
	def __init__(self, 
				conn=None,
				chunk: int=200,
				text_queue=None,
				shutdown_event=None,
				input_files: List=[],
				interrupt_stop_event=None,
				is_transform_data: bool=False,
				):
		threading.Thread.__init__(self)
		set_log_file(file_name="data_worker")
		self.conn = conn
		self.chunk = chunk
		self.text_queue = text_queue
		self.input_files = input_files
		self.shutdown_event = shutdown_event
		self.is_transform_data = is_transform_data
		self.interrupt_stop_event = interrupt_stop_event

		self.time_sleep = 0.02
		self.max_queue_size = 5000//self.chunk
		self.data_queue = queue.Queue()

	def poll_connection(self):
		while not self.shutdown_event.is_set():
			if self.conn.poll(0.01):    #This will return a boolean as to whether there is data to be received and read from the pipe
				try:
					data = self.conn.recv()
					# print(f"----data: {data}")
					self.data_queue.put(data)
				except Exception as e:
					logger.error(f"Error receiving data from connection: {e}")
			else:
				time.sleep(self.time_sleep)

	def split_text(self, text):
		text_splited = text.split()
		texts = []
		# print(f"----len(text_splited): {len(text_splited)}")
		while len(text_splited) > self.chunk:
			texts.append(" ".join(text_splited[:self.chunk]))
			text_splited = text_splited[self.chunk:]
		if text_splited:
			texts.append(" ".join(text_splited))
		return texts

	@staticmethod
	def nat_normalize_text(text):
		text = unicodedata.normalize("NFKC", text)
		text = text.lower().strip()
		sil = FLAGS.special_phonemes[FLAGS.sil_index]
		text = re.sub(r"[\n.,:]+", f" {sil} ", text)
		text = text.replace('"', " ")
		text = re.sub(r"\s+", " ", text)
		text = re.sub(r"[.,:;?!]+", f" {sil} ", text)
		text = re.sub("[ ]+", " ", text)
		text = re.sub(f"( {sil}+)+ ", f" {sil} ", text)
		return text.strip()

	@staticmethod
	def fast_normalize_text(text):
		text = text.replace("\n", " ")
		return text.strip()

	def run(self,):
		if not self.input_files:
			# Start the polling thread
			polling_thread = threading.Thread(target=self.poll_connection)
			polling_thread.start()

		try:
			while not self.shutdown_event.is_set():
				try:
					if self.input_files:
						# datas = []
						for inpf in self.input_files:
							if not os.path.exists(inpf):
								logger.error(f"File {inpf} doesn't exists")
								continue
							inp_text = ""
							if inpf.endswith((".txt")):
								with open(inpf) as file:
									data = file.readlines()
								for line in data:
									inp_text += line
							if self.is_transform_data:
								processed_data = self.nat_normalize_text(inp_text)
								processed_data = self.split_text(processed_data)
							else:
								processed_data = inp_text
								processed_data = self.split_text(processed_data)
							# datas.append(inp_text)
							for i, pdt in enumerate(processed_data):
								if i==0:
									self.text_queue.put((pdt,0))
								else:
									self.text_queue.put((pdt,1))
							print(f"----Size queue: {self.text_queue.qsize()}")
							if self.text_queue.qsize() > self.max_queue_size:
								while self.text_queue.qsize() > self.max_queue_size:
									time.sleep(1)
						# if not datas:
						# 	logger.error(f"All files do not exist")
						break
					else:
						try:
							data = self.data_queue.get(timeout=0.1)
							if self.is_transform_data:
								processed_data = self.nat_normalize_text(data)
								processed_data = self.split_text(processed_data)
							else:
								processed_data = data
								processed_data = self.split_text(processed_data)
							for i, pdt in enumerate(processed_data):
								if i==0:
									self.text_queue.put((pdt,0))
								else:
									self.text_queue.put((pdt,1))
						except queue.Empty:
							continue

				except Exception as e:
					logger.error(f"Unknown error during getting data: {e}")
					tb_str = traceback.format_exc()
					logger.error(f"Traceback: {tb_str}")
					break

		except KeyboardInterrupt:
			self.interrupt_stop_event.set()
			logger.debug("Data worker process finished due to KeyboardInterrupt")
		finally:
			if not self.input_files:
				polling_thread.join()  # Wait for the polling thread to finish

if __name__ == '__main__':
	input_files = ["./data_test/transcript.txt"]
	chunk = 200
	text_queue = mp.Queue()
	shutdown_event = mp.Event()
	interrupt_stop_event = mp.Event()
	parent_data_pipe, child_data_pipe = mp.Pipe()
	is_transform_data = False

	dtw = DataWorker(child_data_pipe,
					chunk,
					text_queue,
					shutdown_event,
					input_files,
					interrupt_stop_event,
					is_transform_data)
	dtw.daemon = True
	dtw.start()

	data_test = []
	with open("./data_test/transcript.txt") as file:
		data = file.readlines()
	for line in data:
		data_test.append(line)
	data_test = [" ".join(data_test)]
	i = len(data_test)
	print(i)
	while True:
		try:
			for dt in data_test:
				parent_data_pipe.send(dt)
			data_processed, cont = dtw.text_queue.get(timeout=0.1)
			print(data_processed)
			print(cont)
			while not dtw.text_queue.empty():
				dtw.text_queue.get_nowait()
		except queue.Empty:
			continue