import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
DIR = FILE.parents[0]
if DIR not in sys.path:
	sys.path.append(str(DIR))

import time
import queue
import torch
import threading
import traceback
import multiprocessing as mp
from base.config import logger, set_log_file
from typing import Iterable, List, Optional, Union
import numpy as np
import json
import onnxruntime
from enum import Enum

from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run

class PhonemeType(str, Enum):
	ESPEAK = "espeak"
	TEXT = "text"

class SynthesisWorker(threading.Thread):
	def __init__(self, 
				conn=None, 
				device: str = "cuda",
				ready_event=None, 
				shutdown_event=None, 
				interrupt_stop_event=None, 
				model_path: str=f"{DIR}/weights/vits_vi.onnx",
				config_path: str=f"{DIR}/weights/vits_vi.json",
				speed: str=f"normal",
				):
		threading.Thread.__init__(self)
		set_log_file(file_name="synthesis_worker")
		self.conn = conn
		self.ready_event = ready_event
		self.shutdown_event = shutdown_event
		self.interrupt_stop_event = interrupt_stop_event

		self.time_sleep = 0.02
		self.noise_scale_w = 0.8
		self.noise_scale = 0.667
		self.queue = queue.Queue()
		self.pad = "_"  # padding (0)
		self.eos = "$"  # end of sentence
		self.bos = "^"  # beginning of sentence
		self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

		speed_values = {"very_slow":1.5,
						"slow":1.2,
						"normal":1,
						"fast":0.6,
						"very_fast":0.4}
		self.speed = float(speed_values[speed.strip()])

		# model
		try:
			sess_options = onnxruntime.SessionOptions()
			self.model = onnxruntime.InferenceSession(model_path, sess_options=sess_options)
			with open(config_path, "r") as file:
				self.config = json.load(file)
		except Exception as e:
			logger.exception(f"Error initializing vits model: {e}")
			raise

		self.ready_event.set()
		logger.debug("Synthesis worker initialized successfully")

	def poll_connection(self):
		while not self.shutdown_event.is_set():
			if self.conn.poll(0.01):    #This will return a boolean as to whether there is data to be received and read from the pipe
				try:
					data, cont = self.conn.recv()
					# print(f"----data: {data}")
					self.queue.put((data, cont))
				except Exception as e:
					logger.error(f"Error receiving data from connection: {e}")
			else:
				time.sleep(self.time_sleep)

	def phonemize(self, text: str) -> List[List[str]]:
		"""Text to phonemes grouped by sentence."""
		if self.config["phoneme_type"] == PhonemeType.ESPEAK:
			if self.config["espeak"]["voice"] == "ar":
				# Arabic diacritization
				# https://github.com/mush42/libtashkeel/
				text = tashkeel_run(text)
			return phonemize_espeak(text, self.config["espeak"]["voice"])
		if self.config["phoneme_type"] == PhonemeType.TEXT:
			return phonemize_codepoints(text)
		logger.error(f'Unexpected phoneme type: {self.config["phoneme_type"]}')
		tb_str = traceback.format_exc()
		print(tb_str)
		# raise ValueError(f'Unexpected phoneme type: {self.config["phoneme_type"]}')

	def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
		"""Phonemes to ids."""
		id_map = self.config["phoneme_id_map"]
		ids: List[int] = list(id_map[self.bos])
		for phoneme in phonemes:
			if phoneme not in id_map:
				print("Missing phoneme from id map: %s", phoneme)
				continue
			ids.extend(id_map[phoneme])
			ids.extend(id_map[self.pad])
		ids.extend(id_map[self.eos])
		return ids

	def audio_float_to_int16(self,
							audio: np.ndarray, 
							max_wav_value: float = 32767.0,
							) -> np.ndarray:
		"""Normalize audio and convert to int16 range"""
		audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
		audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
		audio_norm = audio_norm.astype("int16")
		return audio_norm

	def text2speech(self, text):
		text = text.strip()
		phonemes_list = self.phonemize(text)
		phoneme_ids = []
		for phonemes in phonemes_list:
			phoneme_ids.append(self.phonemes_to_ids(phonemes))

		speaker_id = None
		phoneme_ids_flatten = []
		for i in phoneme_ids:
			phoneme_ids_flatten += i + [0,0,0]
		text = np.expand_dims(np.array(phoneme_ids_flatten, dtype=np.int64), 0)
		text_lengths = np.array([text.shape[1]], dtype=np.int64)
		scales = np.array(
			[self.noise_scale, self.speed, self.noise_scale_w],
			dtype=np.float32,
		)
		sid = None

		if speaker_id is not None:
			sid = np.array([speaker_id], dtype=np.int64)

		start_time = time.perf_counter()
		audio = self.model.run(
			None,
			{
				"input": text,
				"input_lengths": text_lengths,
				"scales": scales,
				"sid": sid,
			},
		)[0].squeeze((0, 1))
		audio = self.audio_float_to_int16(audio.squeeze())
		return audio

	def run(self):
		# Start the polling thread
		polling_thread = threading.Thread(target=self.poll_connection)
		polling_thread.start()

		try:
			while not self.shutdown_event.is_set():
				try:
					st_time = time.time()
					text, cont = self.queue.get(timeout=0.1)
					print(text)
					wave = self.text2speech(text)
					self.conn.send(('success', (wave, cont)))
					print(f"----Duration: {time.time()-st_time}")
				except queue.Empty:
					continue
				except KeyboardInterrupt:
					self.interrupt_stop_event.set()
					logger.debug("Synthesis worker process finished due to KeyboardInterrupt")
					break
				except Exception as e:
					logger.error(f"Unknow error in process of Synthesis worker: {e}")
					tb_str = traceback.format_exc()
					logger.error(f"Traceback: {tb_str}")
					self.conn.send(('error', str(e)))
		finally:
			self.conn.close()
			self.shutdown_event.set()  # Ensure the polling thread will stop
			polling_thread.join()  # Wait for the polling thread to finish

if __name__=="__main__":
	parent_synthesis_pipe, child_synthesis_pipe = mp.Pipe()
	device = "cuda"
	ready_event = mp.Event()
	shutdown_event = mp.Event()
	interrupt_stop_event = mp.Event()
	model_path = f"{DIR}/weights/vits_vi.onnx"
	config_path = f"{DIR}/weights/vits_vi.json"
	speed = "normal"
	syntw = SynthesisWorker(child_synthesis_pipe,
							device,
							ready_event,
							shutdown_event,
							interrupt_stop_event,
							model_path,
							config_path,
							speed)
	syntw.daemon = True
	syntw.start()
	# text = "Quê hương anh nước mặn, đồng chua. Làng tôi nghèo đất cày lên sỏi đá"
	# # audio = syntw.text2speech(text)
	# # print(audio)
	# parent_synthesis_pipe.send(text)
	# if parent_synthesis_pipe.poll(timeout=10):  # Wait for 5 seconds
	# 	status, wave = parent_synthesis_pipe.recv()
	# 	print(f"----status: {status}")

	# sample_rate = 22050
	# from wavfile import write as write_wav
	# write_wav("test.wav", sample_rate, wave)
	# exit()
	#---------------get data---------------
	from data_worker import DataWorker
	import soundfile as sf
	sample_rate = 22050
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
	#/////////////////////////////////////

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
			parent_synthesis_pipe.send((data_processed, cont))
			if parent_synthesis_pipe.poll(timeout=6):  # Wait for 5 seconds
				status, result = parent_synthesis_pipe.recv()
				wave, cont = result
				print(f"----status: {status}")
				print(cont)
				sf.write(f"./outputs/test_999.wav", wave, samplerate=sample_rate)
				i -= 1
				if i<1:
					break
		except queue.Empty:
			continue