import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
DIR = FILE.parents[0]
if DIR not in sys.path:
	sys.path.append(str(DIR))

from typing import Iterable, List, Optional, Union
import multiprocessing as mp
import threading
import halo
import traceback
import queue
import time
import numpy as np

from base.config import logger, set_log_file
from data_worker import DataWorker
from synthesis_worker_vits import SynthesisWorker

class T2A:
	def __init__(self, 
				spinner=True,
				#----data worker----
				chunk: int = 200,
				input_files: List = [],
				is_transform_data: bool = False,

				#----synthesis worker----
				device: str = "cuda",
				speed: str = f"normal",
				model_path: str = f"{DIR}/weights/vits_vi.onnx",
				config_path: str = f"{DIR}/weights/vits_vi.json",
				):
		set_log_file(file_name="app")
		self.halo = None
		self.num_tail = 0
		self.spinner = spinner
		self.num_try_get_result = 0
		# flags controller
		self.is_running = True 
		self.state = "inactive"
		self.is_new_text = False
		self.is_shut_down = False 
		self.is_waiting_text = True
		self.is_synthersizing = False 

		self.ready_event = mp.Event()
		self.shutdown_event = mp.Event()
		self.shutdown_lock = threading.Lock()
		self.interrupt_stop_event = mp.Event()
		self.get_output_lock = threading.Lock()

		#----Start data reading process
		self.input_files = input_files
		self.chunk = chunk
		self.text_queue = mp.Queue()
		self.parent_data_pipe, child_data_pipe = mp.Pipe()
		self.is_transform_data = is_transform_data
		self.dtw = DataWorker(child_data_pipe,
							self.chunk,
							self.text_queue,
							self.shutdown_event,
							self.input_files,
							self.interrupt_stop_event,
							self.is_transform_data)
		self.dtw.daemon = True
		self.dtw.start()

		#----Start synthesis worker----
		self.parent_synthesis_pipe, child_synthesis_pipe = mp.Pipe()
		self.device = device
		self.model_path = model_path
		self.config_path = config_path
		self.speed = speed
		self.syntw = SynthesisWorker(child_synthesis_pipe,
									self.device,
									self.ready_event,
									self.shutdown_event,
									self.interrupt_stop_event,
									self.model_path,
									self.config_path,
									self.speed)
		self.syntw.daemon = True
		self.syntw.start()

		if input_files:
			self.rfile_thread = threading.Thread(target=self._execute_file)
			self.rfile_thread.daemon = True
			self.rfile_thread.start()
		else:
			self.exe_thread = threading.Thread(target=self._execute_text)
			self.exe_thread.daemon = True
			self.exe_thread.start()

		# Wait for all models to start
		logger.debug('Waiting for main transcription model to start')
		self.ready_event.wait()
		logger.info('Main process is ready')


	def start(self,):
		self.is_waiting_text = True
		self.is_synthersizing = False
		self.num_tail = 0
		self.is_new_text = False

	def stop(self,):
		self.is_waiting_text = False
		self.is_synthersizing = True

	def shutdown(self,):
		with self.shutdown_lock:
			if self.is_shut_down:
				return
			logger.info("T2A shutting down!")

			self.shutdown_event.set()
			self.is_running = False 
			self.is_shut_down = True
			self.parent_data_pipe.close()
			self.parent_synthesis_pipe.close()
			self.interrupt_stop_event.set()

			logger.info("Finishing threads")
			if self.exe_thread:
				self.exe_thread.join()

			if self.rfile_thread:
				self.rfile_thread.join()

	def send_data(self, text):
		if not self.is_running:
			logger.error("Program stoped, please don't send data anymore and start the program again")
			return 
		try:
			self.parent_data_pipe.send(text)
		except BrokenPipeError:
			logger.error("BrokenPipeError _recording_worker")
			self.is_running = False
		except Exception as e:
			logger.error(f"Internal error of send_data: {e}")
			self.is_running = False
		return

	def _execute_text(self,):
		while self.is_running:
			if self.is_waiting_text and not self.is_synthersizing:
				self._set_state("listening")
				while not self.dtw.text_queue.empty():
					self.dtw.text_queue.get_nowait()
			try:
				try:
					data_processed, cont = self.dtw.text_queue.get(timeout=0.01)
				except queue.Empty:
					if not self.is_running:
						logger.info("----Not running, breaking loop")
						break
					continue
			except BrokenPipeError:
				logger.error("BrokenPipeError _recording_worker")
				self.is_running = False
				break

			if self.is_synthersizing:
				self._set_state("transcribing")
				if cont and not self.is_new_text:
					self.num_tail += 1
				elif not cont:
					self.is_new_text = True
				elif self.is_new_text:
					continue

			# if self.is_new_text:
			# 	self.parent_synthesis_pipe.send((data_processed, 0))
			# 	self.stop()
			# else:
			# 	if self.is_synthersizing:
			# 		self._set_state("transcribing")

			self.parent_synthesis_pipe.send((data_processed, cont))
			self._set_state("transcribing")
			self.stop()


	def _execute_file(self,):
		while self.is_running:
			if self.is_waiting_text and not self.is_synthersizing:
				self._set_state("listening")

			try:
				try:
					data_processed, cont = self.dtw.text_queue.get(timeout=0.01)
				except queue.Empty:
					if not self.is_running:
						logger.info("----Not running, breaking loop")
						break
					continue
			except BrokenPipeError:
				logger.error("BrokenPipeError _recording_worker")
				self.is_running = False
				break

			if self.is_synthersizing:
				self._set_state("transcribing")
				if cont and not self.is_new_text:
					self.num_tail += 1
				elif not cont:
					self.is_new_text = True
				elif self.is_new_text:
					logger.info("Model is transcibing the previous text, please waiting until the model finishs")
					# continue
					while self.num_tail:
						time.sleep(1)

			self.parent_synthesis_pipe.send((data_processed, cont))
			self._set_state("transcribing")
			self.stop()

	def get_sound(self,):
		with self.get_output_lock:
			while True:
				if not self.is_running:
					break
				if self.is_shut_down or self.interrupt_stop_event.is_set():
					yield ""
					break
				try:
					if self.parent_synthesis_pipe.poll(timeout=5):  # Wait for 5 seconds
						status, result = self.parent_synthesis_pipe.recv()
						print("----status: ", status)
						if status == "success":
							wave, cont = result
							if not self.num_tail:
								yield (wave, cont)
								break
							else:
								self.num_tail -= 1
								yield (wave, cont)

					else: 
						time.sleep(1)
						# print("waiting result....")
						self.num_try_get_result += 1
						if self.num_try_get_result>1:
							self.num_try_get_result = 0
							break
						continue
				except KeyboardInterrupt:
					logger.error("KeyboardInterrupt in get_sound() method")
					self.shutdown()
					# yield 
					return None
				except Exception as e:
					logger.error(f"Error during receiving transcription: {str(e)}")
					tb_str = traceback.format_exc()
					logger.error(f"Traceback: {tb_str}")
					self.shutdown()
					yield None
					return 
			self.start()
			yield None
			# return None

	def _set_state(self, new_state): 
		"""
		Update the current state of the recorder and execute
		corresponding state-change callbacks.

		Args:
			new_state (str): The new state to set.

		"""
		# Check if the state has actually changed
		if new_state == self.state:
			return

		# Store the current state for later comparison
		old_state = self.state

		# Update to the new state
		self.state = new_state

		# Log the state change
		logger.info(f"State changed from '{old_state}' to '{new_state}'")

		# # Execute callbacks based on transitioning FROM a particular state
		# if old_state == "listening":
		# 	if self.on_vad_detect_stop:
		# 		self.on_vad_detect_stop()
		# elif old_state == "wakeword":
		# 	if self.on_wakeword_detection_end:
		# 		self.on_wakeword_detection_end()

		# Execute callbacks based on transitioning TO a particular state
		if new_state == "listening":
			# if self.on_vad_detect_start:
			# 	self.on_vad_detect_start()
			self._set_spinner("waiting text")
			if self.spinner and self.halo:
				self.halo._interval = 200
		elif new_state == "transcribing":
			# if self.on_transcription_start:
			# 	self.on_transcription_start()
			self._set_spinner("transcribing")
			if self.spinner and self.halo:
				self.halo._interval = 300 	#50
		elif new_state == "recording":
			self._set_spinner("recording")
			if self.spinner and self.halo:
				self.halo._interval = 100
		elif new_state == "inactive":
			if self.spinner and self.halo:
				self.halo.stop()
				self.halo = None
		elif new_state == "reading":
			self._set_spinner("reading file")
			if self.spinner and self.halo:
				self.halo._interval = 100

	def _set_spinner(self, text):
		"""
		Update the spinner's text or create a new
		spinner with the provided text.

		Args:
			text (str): The text to be displayed alongside the spinner.
		"""
		if self.spinner:
			# If the Halo spinner doesn't exist, create and start it
			if self.halo is None:
				self.halo = halo.Halo(text=text)
				self.halo.start()
			# If the Halo spinner already exists, just update the text
			else:
				self.halo.text = text

if __name__=="__main__":
	t2a = T2A(input_files=["./data_test/transcript.txt"])

	data_test = []
	with open("./data_test/transcript.txt") as file:
		data = file.readlines()
	for line in data:
		data_test.append(line)

	data_test = [" ".join(data_test)]
	print(f"----data_test: {data_test}")
	for dt in data_test:
		t2a.send_data(dt)

	outputs = []
	import soundfile as sf
	while True:
		result = t2a.get_sound()
		print(result)
		re = next(result)
		print(re)
		if re:
			wave, cont = re
			print(cont)
			print(wave.dtype)
			print(f"----wave: {wave}")
			if not cont:
				outputs.clear()
				outputs.extend(wave.tolist())
			else:
				outputs.extend(wave.tolist())
			
		else:
			if outputs:
				sf.write(f"test2.wav", np.array(outputs, dtype=np.int16), samplerate=22050)
			# break
			continue