"""
core_voice_typing.py
====================

Backend do “Digitador por Voz” refatorado com base nos 7 tópicos
de arquitetura. Não contém interface gráfica – isso ficará na Parte 2.

Principais componentes
----------------------
• AppState / StateManager ........... FSM de 4 estados (Inativo, Pronto, Gravando, Pausado)
• VoiceTypingEngine ................. Orquestra Recorder, Transcriber e Typer
• Recorder .......................... Captura áudio do microfone e coloca em fila
• Transcriber ....................... Usa Whisper para transformar áudio em texto
• Typer ............................. Digita texto no cursor atual
• KeyboardListener .................. Escuta atalhos globais F8 / F9 e aciona StateManager
"""

from __future__ import annotations

import os
import sys
import time
import wave
import queue
import tempfile
import threading
import unicodedata
from enum import Enum, auto
from datetime import datetime
from typing import Callable, List, Optional

# ====== dependências externas ======
import numpy as np
import pyaudio
import keyboard          # Atalhos globais
import pyautogui         # Digitação
import torch             # CUDA check
import whisper           # Modelo
# ===================================


# ---------------------------------------------------------------------------
# 1. Estados da aplicação
# ---------------------------------------------------------------------------

class AppState(Enum):
    INACTIVE = auto()   # Aplicação desligada
    READY    = auto()   # Ligada, aguardando gravação
    RECORDING = auto()  # Capturando áudio
    PAUSED    = auto()  # Ligada, mas captura pausada


# ---------------------------------------------------------------------------
# 2. StateManager – Finite‑State‑Machine
# ---------------------------------------------------------------------------

class StateManager:
    """Controla o estado global da aplicação e notifica ouvintes."""

    def __init__(self) -> None:
        self._state = AppState.INACTIVE
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[AppState], None]] = []

    # ---- Observers --------------------------------------------------------
    def on_state_change(self, callback: Callable[[AppState], None]) -> None:
        """Registra callback para receber atualizações de estado."""
        self._callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._callbacks:
            try:
                cb(self._state)
            except Exception:
                pass  # Nunca quebra a FSM

    # ---- getters ----------------------------------------------------------
    @property
    def state(self) -> AppState:
        return self._state

    # ---- transições -------------------------------------------------------
    def activate(self) -> None:
        with self._lock:
            if self._state is AppState.INACTIVE:
                self._state = AppState.READY
                self._notify()

    def deactivate(self) -> None:
        with self._lock:
            if self._state in (AppState.READY, AppState.PAUSED, AppState.RECORDING):
                self._state = AppState.INACTIVE
                self._notify()

    def start_recording(self) -> None:
        with self._lock:
            if self._state in (AppState.READY, AppState.PAUSED):
                self._state = AppState.RECORDING
                self._notify()

    def pause_recording(self) -> None:
        with self._lock:
            if self._state is AppState.RECORDING:
                self._state = AppState.PAUSED
                self._notify()

    # Utilidade para encerrar aplicação limpo
    def stop_app(self) -> None:
        with self._lock:
            self._state = AppState.INACTIVE
            self._notify()


# ---------------------------------------------------------------------------
# 3. Recorder – captura de microfone  ✅ FIXED
# ---------------------------------------------------------------------------

class Recorder(threading.Thread):
    """
    Mantém‑se vivo todo o tempo em que o app está ativo.
    Lê áudio somente quando o estado é RECORDING.
    """

    def __init__(
        self,
        state_manager: StateManager,
        audio_queue: queue.Queue[str],
        sample_rate: int = 48_000,
        chunk_size: int = 1024,
        silence_threshold: int = 300,
        silence_duration: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self.state_manager = state_manager
        self.audio_queue = audio_queue

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.required_silent_chunks = int(silence_duration * sample_rate / chunk_size)

        self._temp_dir = tempfile.mkdtemp()
        self._terminate = threading.Event()
        self._recording = threading.Event()   # ← novo: indica se deve capturar

        # Listener para mudanças de estado
        self.state_manager.on_state_change(self._handle_state_change)

        # Sobe a thread já na criação; ela fica em espera até _recording.set()
        self.start()

    # ------------------ life‑cycle ------------------
    def _handle_state_change(self, state: AppState) -> None:
        if state is AppState.RECORDING:
            self._recording.set()          # COMEÇAR a capturar
        else:
            self._recording.clear()        # Parar captura (mas mantém thread viva)

        if state is AppState.INACTIVE:
            self._terminate.set()          # Encerra definitivamente

    def run(self) -> None:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames: List[bytes] = []
        silent_chunks = 0

        while not self._terminate.is_set():
            if not self._recording.is_set():
                time.sleep(0.05)
                continue  # apenas espera até alguém pedir gravação

            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)

            # Detecção de silêncio
            audio_data = np.frombuffer(data, dtype=np.int16)
            silent_chunks = (
                silent_chunks + 1
                if np.abs(audio_data).mean() < self.silence_threshold
                else 0
            )

            if silent_chunks >= self.required_silent_chunks and len(frames) > silent_chunks:
                segment = frames[:-silent_chunks]
                if segment:
                    file_path = self._write_wav(segment)
                    self.audio_queue.put(file_path)
                frames = frames[-silent_chunks:]

        # Flush final
        if frames:
            file_path = self._write_wav(frames)
            self.audio_queue.put(file_path)

        stream.stop_stream()
        stream.close()
        p.terminate()

    # ------------------ helpers ------------------
    def _write_wav(self, frames: List[bytes]) -> str:
        file_path = os.path.join(self._temp_dir, f"segment_{time.time():.0f}.wav")
        wf = wave.open(file_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        return file_path


# ---------------------------------------------------------------------------
# 4. Transcriber – Whisper + CUDA  ✅ FIXED
# ---------------------------------------------------------------------------

class Transcriber(threading.Thread):
    """Mantém‑se viva; transcreve quando há arquivos na fila e estamos gravando."""

    def __init__(
        self,
        state_manager: StateManager,
        audio_queue: queue.Queue[str],
        output_callback: Callable[[str], None],
        model_size: str = "medium",
        language: str = "pt",
    ) -> None:
        super().__init__(daemon=True)
        self.state_manager = state_manager
        self.audio_queue = audio_queue
        self.output_callback = output_callback

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)

        self._terminate = threading.Event()
        self._process_allowed = threading.Event()   # gravação ligada?

        self.state_manager.on_state_change(self._handle_state_change)
        self.start()

    def _handle_state_change(self, state: AppState) -> None:
        if state is AppState.RECORDING:
            self._process_allowed.set()
        else:
            self._process_allowed.clear()

        if state is AppState.INACTIVE:
            self._terminate.set()

    def run(self) -> None:
        while not self._terminate.is_set():
            if not self._process_allowed.is_set():
                time.sleep(0.05)
                continue

            try:
                audio_file = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self.model.transcribe(
                    audio_file,
                    language="pt",
                    fp16=torch.cuda.is_available(),
                )
                text = result["text"].strip()
                if text:
                    self.output_callback(text)
            finally:
                try:
                    os.remove(audio_file)
                except FileNotFoundError:
                    pass


# ---------------------------------------------------------------------------
# 5. Typer – digitação robusta
# ---------------------------------------------------------------------------

class Typer:
    """Envia texto ao cursor atual (keyboard.write com fallback)."""

    @staticmethod
    def type_text(text: str) -> None:
        try:
            keyboard.write(text + " ")
        except Exception:
            # Fallback caracter‑a‑caracter, inspirado no código legado :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
            for ch in text:
                if ch in "?!.:()[]{}\"'":
                    pyautogui.write(ch)
                elif unicodedata.category(ch).startswith("L"):
                    pyautogui.write(ch)
                else:
                    try:
                        pyautogui.press(ch)
                    except Exception:
                        pass
            pyautogui.write(" ")


# ---------------------------------------------------------------------------
# 6. VoiceTypingEngine – orquestra tudo
# ---------------------------------------------------------------------------

class VoiceTypingEngine:
    """
    Casca única que une Recorder + Transcriber + Typer e conversa com a FSM.
    A GUI (Parte 2) só precisará instanciar este engine e observar eventos.
    """

    def __init__(
        self,
        state_manager: StateManager,
        model_size: str = "medium",
        language: str = "pt",
    ) -> None:
        self.state_manager = state_manager
        self.audio_queue: "queue.Queue[str]" = queue.Queue()

        # Componentes
        self.recorder = Recorder(state_manager, self.audio_queue)
        self.transcriber = Transcriber(
            state_manager,
            self.audio_queue,
            output_callback=self._handle_text,
            model_size=model_size,
            language=language,
        )
        self.typer = Typer()

        # Log / callbacks externos
        self._status_callbacks: List[Callable[[str], None]] = []

    # ------------------ APIs para GUI ------------------
    def add_status_callback(self, cb: Callable[[str], None]) -> None:
        self._status_callbacks.append(cb)

    def _update_status(self, msg: str) -> None:
        for cb in self._status_callbacks:
            try:
                cb(msg)
            except Exception:
                pass

    # ------------------ Out‑pipeline ------------------
    def _handle_text(self, text: str) -> None:
        """Recebe texto transcrito → digita → notifica status."""
        self.typer.type_text(text)
        self._update_status(f"Transcrito: «{text}»")

    # ------------------ Exposed control ----------------
    def shutdown(self) -> None:
        """Usado ao fechar o aplicativo por completo."""
        self.state_manager.stop_app()


# ---------------------------------------------------------------------------
# 7. KeyboardListener – atalhos F8/F9
# ---------------------------------------------------------------------------

class KeyboardListener(threading.Thread):
    """
    Thread que escuta globalmente F8 (start) e F9 (pause).
    Sobe e desce junto com o estado READY / INACTIVE.
    """

    def __init__(self, state_manager: StateManager) -> None:
        super().__init__(daemon=True)
        self.state_manager = state_manager
        self._should_stop = threading.Event()
        self.state_manager.on_state_change(self._handle_state_change)

    def _handle_state_change(self, state: AppState) -> None:
        if state is AppState.READY and not self.is_alive():
            self.start()
        elif state is AppState.INACTIVE:
            self._should_stop.set()

    def run(self) -> None:
        keyboard.add_hotkey("F8", lambda: self.state_manager.start_recording())
        keyboard.add_hotkey("F9", lambda: self.state_manager.pause_recording())

        while not self._should_stop.is_set():
            time.sleep(0.1)

        keyboard.remove_hotkey("F8")
        keyboard.remove_hotkey("F9")


# ---------------------------------------------------------------------------
# 8. Uso rápido (ex.: testes sem GUI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Rodar este arquivo direto faz um teste de linha‑de‑comando:

    - Pressione Enter para ativar (equivalente ao botão Iniciar da GUI)
    - Use F8 e F9 como na especificação
    - Ctrl+C encerra
    """
    print(">> MODO TESTE – Backend apenas")
    sm = StateManager()
    engine = VoiceTypingEngine(sm)
    kl = KeyboardListener(sm)

    input("Pressione ENTER para ativar o aplicativo... ")
    sm.activate()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nEncerrando…")
    finally:
        engine.shutdown()
