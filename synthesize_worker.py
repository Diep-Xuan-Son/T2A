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
import haiku as hk
import jax
import pickle
import jax.numpy as jnp
import numpy as np
import json

from nat.data_loader import load_phonemes_set
from nat.config import FLAGS, DurationInput
from nat.model import AcousticModel, DurationModel
from hifigan.model import Generator

class SynthesisWorker(threading.Thread):
	def __init__(self, 
				conn=None, 
				device: str = "cuda",
				ready_event=None, 
				shutdown_event=None, 
				interrupt_stop_event=None, 
				# text to token
				lexicon_fn: str=f"{DIR}/weights/nat/lexicon.txt",
				# duration model (this model for generate human-like rhythms of speech)
				model_duration: str=f"{DIR}/weights/nat/duration_latest_ckpt.pickle",
				# text2mel model
				model_t2m: str=f"{DIR}/weights/nat/acoustic_latest_ckpt.pickle",
				# mel2wav model
				model_mel2wav: str=f"{DIR}/weights/hifigan/hk_hifi.pickle",
				config_mel2wav: str=f"{DIR}/weights/hifigan/config.json",
				):
		threading.Thread.__init__(self)
		set_log_file(file_name="synthesis_worker")
		self.conn = conn
		self.model_t2m = model_t2m
		self.lexicon_fn = lexicon_fn
		self.ready_event = ready_event
		self.model_mel2wav = model_mel2wav
		self.shutdown_event = shutdown_event
		self.model_duration = model_duration
		self.config_mel2wav = config_mel2wav
		self.interrupt_stop_event = interrupt_stop_event

		self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
		self.time_sleep = 0.02
		self.queue = queue.Queue()

		# duration model
		try:
			self.forward_fn = jax.jit(hk.transform_with_state(self.fwd_).apply)
			with open(self.model_duration, "rb") as f:
				self.dic = pickle.load(f)
		except Exception as e:
			logger.exception(f"Error initializing duration model: {e}")
			raise

		# model predict mel
		try:
			self.predict_fn = jax.jit(self.forward.apply, static_argnums=[5])
			with open(self.model_t2m, "rb") as f:
				dic = pickle.load(f)
				self.last_step, self.params, self.aux, self.rng, self.optim_state = (
						dic["step"],
						dic["params"],
						dic["aux"],
						dic["rng"],
						dic["optim_state"],
				)
		except Exception as e:
			logger.exception(f"Error initializing mel prediction model: {e}")
			raise

		# model mel2wave
		# with open(self.config_mel2wav) as f:
		# 	data = f.read()
		# json_config = json.loads(data)
		# self.h = AttrDict(json_config)
		try:
			self.rng = next(hk.PRNGSequence(42))
			with open(self.model_mel2wav, "rb") as f:
				self.params_hifi = pickle.load(f)
		except Exception as e:
			logger.exception(f"Error initializing mel2wav model: {e}")
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
					logging.error(f"Error receiving data from connection: {e}")
			else:
				time.sleep(self.time_sleep)

	@staticmethod
	def fwd_(x):
		return DurationModel(is_training=False)(x)

	@staticmethod
	@hk.transform_with_state
	def forward(tokens, durations, n_frames):
		net = AcousticModel(is_training=False)
		return net.inference(tokens, durations, n_frames)

	@staticmethod
	@hk.transform_with_state
	def forward_hifi(x):
		# with open(f"{DIR}/weights/hifigan/config.json") as f:
		# 	data = f.read()
		# json_config = json.loads(data)
		# h = AttrDict(json_config)
		net = Generator()
		return net(x)

	@staticmethod
	def load_lexicon(fn):
		lines = open(fn, "r").readlines()
		lines = [l.lower().strip().split("\t") for l in lines]
		return dict(lines)

	def text2tokens(self, text):
		phonemes = load_phonemes_set()
		lexicon = self.load_lexicon(self.lexicon_fn)

		words = text.strip().lower().split()
		tokens = [FLAGS.sil_index]
		for word in words:
			if word in FLAGS.special_phonemes:
				tokens.append(phonemes.index(word))
			elif word in lexicon:
				p = lexicon[word]
				p = p.split()
				p = [phonemes.index(pp) for pp in p]
				tokens.extend(p)
				tokens.append(FLAGS.word_end_index)
			else:
				for p in word:
					if p in phonemes:
						tokens.append(phonemes.index(p))
				tokens.append(FLAGS.word_end_index)
		tokens.append(FLAGS.sil_index)  # silence
		return tokens

	def predict_duration(self, tokens):
		x = DurationInput(
			np.array(tokens, dtype=np.int32)[None, :],
			np.array([len(tokens)], dtype=np.int32),
			None,
		)
		return self.forward_fn(self.dic["params"], self.dic["aux"], self.dic["rng"], x)[0]

	def predict_mel(self, tokens, durations):
		durations = durations * FLAGS.sample_rate / (FLAGS.n_fft // 4)
		n_frames = int(jnp.sum(durations).item())
		tokens = np.array(tokens, dtype=np.int32)[None, :]
		return self.predict_fn(self.params, self.aux, self.rng, tokens, durations, n_frames)[0]

	def text2mel(self, text, silence_duration=-1.0):
		tokens = self.text2tokens(text)
		durations = self.predict_duration(tokens)
		durations = jnp.where(
			np.array(tokens)[None, :] == FLAGS.sil_index,
			jnp.clip(durations, a_min=silence_duration, a_max=None),
			durations,
		)
		durations = jnp.where(
			np.array(tokens)[None, :] == FLAGS.word_end_index, 0.0, durations
		)
		mels = self.predict_mel(tokens, durations)
		if tokens[-1] == FLAGS.sil_index:
			end_silence = durations[0, -1].item()
			silence_frame = int(end_silence * FLAGS.sample_rate / (FLAGS.n_fft // 4))
			mels = mels[:, : (mels.shape[1] - silence_frame)]
		return mels

	def text2mel_full(self, text, silence_duration=-1.0):
		phonemes = load_phonemes_set()
		lexicon = self.load_lexicon(self.lexicon_fn)

		words = text.strip().lower().split()
		tokens = [FLAGS.sil_index]
		for word in words:
			if word in FLAGS.special_phonemes:
				tokens.append(phonemes.index(word))
			elif word in lexicon:
				p = lexicon[word]
				p = p.split()
				p = [phonemes.index(pp) for pp in p]
				tokens.extend(p)
				tokens.append(FLAGS.word_end_index)
			else:
				for p in word:
					if p in phonemes:
						tokens.append(phonemes.index(p))
				tokens.append(FLAGS.word_end_index)
		tokens.append(FLAGS.sil_index)  # silence

		x = DurationInput(
			np.array(tokens, dtype=np.int32)[None, :],
			np.array([len(tokens)], dtype=np.int32),
			None,
		)
		durations = self.forward_fn(self.dic["params"], self.dic["aux"], self.dic["rng"], x)[0]
		durations = jnp.where(
			np.array(tokens)[None, :] == FLAGS.sil_index,
			jnp.clip(durations, a_min=silence_duration, a_max=None),
			durations,
		)
		durations = jnp.where(
			np.array(tokens)[None, :] == FLAGS.word_end_index, 0.0, durations
		)

		durations_transformed = durations * FLAGS.sample_rate / (FLAGS.n_fft // 4)
		n_frames = int(jnp.sum(durations_transformed).item())
		tokens_expand = np.array(tokens, dtype=np.int32)[None, :]
		mels = self.predict_fn(self.params, self.aux, self.rng, tokens_expand, durations_transformed, n_frames)[0]
		if tokens[-1] == FLAGS.sil_index:
			end_silence = durations[0, -1].item()
			silence_frame = int(end_silence * FLAGS.sample_rate / (FLAGS.n_fft // 4))
			mels = mels[:, : (mels.shape[1] - silence_frame)]

		return mels

	def mel2wave(self, mel):
		aux = {}
		print(mel.shape)
		wav, aux = self.forward_hifi.apply(self.params_hifi, aux, self.rng, mel)
		wav = jnp.squeeze(wav)
		audio = jax.device_get(wav)
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
					mel = self.text2mel_full(text)
					print(f"----Duration_text2mel: {time.time()-st_time}")
					wave = self.mel2wave(mel)
					self.conn.send(('success',  (wave, cont)))
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
	# text to token
	lexicon_fn = f"{DIR}/weights/nat/lexicon.txt"
	# duration model (this model for generate human-like rhythms of speech)
	model_duration = f"{DIR}/weights/nat/duration_latest_ckpt.pickle"
	# text2mel model
	model_t2m = f"{DIR}/weights/nat/acoustic_latest_ckpt.pickle"
	# mel2wav model
	model_mel2wav = f"{DIR}/weights/hifigan/hk_hifi.pickle"
	config_mel2wav = f"{DIR}/weights/hifigan/config.json"
	syntw = SynthesisWorker(child_synthesis_pipe,
							device,
							ready_event,
							shutdown_event,
							interrupt_stop_event,
							lexicon_fn,
							model_duration,
							model_t2m,
							model_mel2wav,
							config_mel2wav,)
	syntw.daemon = True
	syntw.start()
	# text = "Quê hương anh nước mặn, đồng chua. Làng tôi nghèo đất cày lên sỏi đá"
	# parent_synthesis_pipe.send(text)
	# if parent_synthesis_pipe.poll(timeout=10):  # Wait for 5 seconds
	# 	status, wave = parent_synthesis_pipe.recv()
	# 	print(f"----status: {status}")
	# import soundfile as sf
	# sample_rate = 16000
	# sf.write("test.wav", wave, samplerate=sample_rate)

	#---------------get data---------------
	from data_worker import DataWorker
	import soundfile as sf
	sample_rate = 16000
	input_files = []
	chunk = 200
	text_queue = mp.Queue()
	shutdown_event = mp.Event()
	interrupt_stop_event = mp.Event()
	parent_data_pipe, child_data_pipe = mp.Pipe()
	dtw = DataWorker(child_data_pipe,
					chunk,
					text_queue,
					shutdown_event,
					input_files,
					interrupt_stop_event)
	dtw.daemon = True
	dtw.start()
	#/////////////////////////////////////

	data_test = []
	with open("./data_test/transcript.txt") as file:
		data = file.readlines()
	for line in data:
		data_test.append(line)

	data_test = [" ".join(data_test)]
	data_test = ["Xin chao viet nam"]
	i = len(data_test)
	print(i)
	while True:
		try:
			for dt in data_test:
				parent_data_pipe.send(dt)
			data_processed, cont = dtw.text_queue.get(timeout=0.1)
			if cont and not buffer:
				buffer.append(data_processed)
				i += 1
			if buffer:
				for b in buffer:
					parent_synthesis_pipe.send((data_processed, 1))
				buffer.clear()
			else:
				parent_synthesis_pipe.send((data_processed, 0))
			if parent_synthesis_pipe.poll(timeout=10):  # Wait for 5 seconds
				status, result = parent_synthesis_pipe.recv()
				wave, cont = result
				print(f"----status: {status}")
				i -= 1
				if i<1:
					break
				sf.write(f"./outputs/test_{i}.wav", wave, samplerate=sample_rate)
		except queue.Empty:
			continue