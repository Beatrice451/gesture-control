import os
import queue
import tempfile
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


class VoiceInput:
    """
    Push-to-talk на Whisper.
    start() / stop() — мгновенные, тяжёлая работа в отдельном потоке.
    Результаты забираются через poll().
    """

    def __init__(
        self,
        model_size="small",
        device="cpu",
        compute_type="int8",
        language=None,
        sample_rate=16000,
        min_duration=0.3,
        initial_prompt=None,
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.language = language
        self.initial_prompt = initial_prompt

        # состояние записи
        self._recording = False
        self._buffer = []
        self._stream = None
        self._lock = threading.Lock()

        # очереди
        self._jobs = queue.Queue()      # путь к wav -> воркер
        self._results = queue.Queue()   # str <- воркер

        # модель (грузится в воркере, чтобы не тормозить старт)
        self._model = None
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type

        self._stop_flag = False
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    # ---------- публичное API ----------

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self):
        """Начать запись. Мгновенно."""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._buffer = []

            def callback(indata, frames, time_info, status):
                # это аудио-поток sounddevice — только копим
                if self._recording:
                    self._buffer.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=callback,
            )
            self._stream.start()

    def stop(self):
        """Остановить запись и отправить на распознавание. Мгновенно."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            stream.stop()
            stream.close()

        with self._lock:
            chunks = self._buffer
            self._buffer = []

        if not chunks:
            return

        audio = np.concatenate(chunks, axis=0)
        if len(audio) / self.sample_rate < self.min_duration:
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, audio, self.sample_rate)
        self._jobs.put(tmp.name)

    def poll(self):
        """Забрать готовые расшифровки, не блокируясь. -> list[str]."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    def shutdown(self):
        """Остановить всё. Быстро, без ожидания Whisper."""
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

        self._stop_flag = True
        self._jobs.put(None)   # разбудить воркер

    # ---------- воркер ----------

    def _run_worker(self):
        # грузим модель лениво, в фоне.
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            print(f"[voice] whisper '{self._model_size}' готов")
        except Exception as e:
            print("[voice] не удалось загрузить модель:", e)
            return

        while not self._stop_flag:
            job = self._jobs.get()
            if job is None:
                break

            try:
                segments, _ = self._model.transcribe(
                    job,
                    language=self.language,
                    vad_filter=True,
                    beam_size=5,
                    initial_prompt=self.initial_prompt,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    self._results.put(text)
            except Exception as e:
                print("[voice] whisper error:", e)
            finally:
                try:
                    os.unlink(job)
                except OSError:
                    pass