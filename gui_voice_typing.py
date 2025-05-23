"""
gui_voice_typing.py
===================

Interface Tkinter que utiliza o backend definido em core_voice_typing.py.

Botões e atalhos
----------------
• Botão GUI “Ativar” .............. liga o aplicativo   (StateManager.activate)
• Botão GUI “Desativar” ........... desliga o app       (StateManager.deactivate)
• Botão GUI “Fechar” .............. encerra programa    (StateManager.stop_app + root.destroy)
• Tecla F8 ........................ inicia gravação     (StateManager.start_recording)
• Tecla F9 ........................ pausa gravação      (StateManager.pause_recording)

Requisitos
----------
Instale as dependências listadas em requirements.txt,
incluindo PyAudio, Tkinter (já vem no Python padrão em Windows/macOS)
e as libs usadas no backend.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox

import torch

from core_voice_typing import (
    AppState,
    StateManager,
    VoiceTypingEngine,
    KeyboardListener,
)


# ---------------------------------------------------------------------------
# 1. Classe principal da GUI
# ---------------------------------------------------------------------------

class VoiceTypingGUI:
    """Janela Tkinter que controla o StateManager e exibe status."""

    REFRESH_MS = 1000  # Atualização de GPU e status 1×/s

    def __init__(self) -> None:
        # Backend ‑‑---------------------------------------------------------
        self.state_manager = StateManager()
        self.engine = VoiceTypingEngine(self.state_manager)
        self.keyboard_listener = KeyboardListener(self.state_manager)

        # GUI ‑‑-------------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("Digitador por Voz – Whisper CUDA")
        self.root.resizable(False, False)

        self._build_widgets()
        self._layout_widgets()
        self._wire_callbacks()

        # Observers para backend
        self.state_manager.on_state_change(self._on_state_change)
        self.engine.add_status_callback(self._on_status)

        # Inicializa tela
        self._on_state_change(self.state_manager.state)
        self._tick()

    # ---------------------------------------------------------------------
    # construção da interface
    # ---------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self.btn_activate = ttk.Button(self.root, text="Ativar (Iniciar)")
        self.btn_deactivate = ttk.Button(self.root, text="Desativar (Pausar)")
        self.btn_close = ttk.Button(self.root, text="Fechar")

        self.lbl_state = ttk.Label(self.root, text="Estado: INATIVO", width=40)
        self.lbl_status = ttk.Label(self.root, text="Pronto.", width=40)
        self.gpu_bar = ttk.Progressbar(
            self.root, orient="horizontal", length=200, mode="determinate"
        )
        self.lbl_gpu = ttk.Label(self.root, text="GPU: n/a")

    def _layout_widgets(self) -> None:
        pad = dict(padx=10, pady=5)
        self.lbl_state.grid(row=0, column=0, columnspan=3, **pad)
        self.lbl_status.grid(row=1, column=0, columnspan=3, **pad)

        self.btn_activate.grid(row=2, column=0, **pad)
        self.btn_deactivate.grid(row=2, column=1, **pad)
        self.btn_close.grid(row=2, column=2, **pad)

        self.gpu_bar.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
        self.lbl_gpu.grid(row=3, column=2, sticky="e", **pad)

    # ---------------------------------------------------------------------
    # callbacks GUI
    # ---------------------------------------------------------------------
    def _wire_callbacks(self) -> None:
        self.btn_activate.config(command=self._on_activate_clicked)
        self.btn_deactivate.config(command=self._on_deactivate_clicked)
        self.btn_close.config(command=self._on_close_clicked)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_clicked)

    def _on_activate_clicked(self) -> None:
        self.state_manager.activate()

    def _on_deactivate_clicked(self) -> None:
        self.state_manager.deactivate()

    def _on_close_clicked(self) -> None:
        if messagebox.askokcancel("Sair", "Deseja fechar o aplicativo?"):
            self.engine.shutdown()
            self.root.destroy()
            sys.exit(0)

    # ---------------------------------------------------------------------
    # Observers do backend (thread‑safe via after)
    # ---------------------------------------------------------------------
    def _on_state_change(self, state: AppState) -> None:
        def update() -> None:
            self.lbl_state.config(text=f"Estado: {state.name}")
            if state is AppState.INACTIVE:
                self.btn_activate.config(state="normal")
                self.btn_deactivate.config(state="disabled")
            elif state in (AppState.READY, AppState.PAUSED):
                self.btn_activate.config(state="disabled")
                self.btn_deactivate.config(state="normal")
            elif state is AppState.RECORDING:
                self.btn_activate.config(state="disabled")
                self.btn_deactivate.config(state="normal")
        self.root.after(0, update)

    def _on_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.lbl_status.config(text=msg))

    # ---------------------------------------------------------------------
    # Atualização periódica (GPU usage etc.)
    # ---------------------------------------------------------------------
    def _tick(self) -> None:
        if torch.cuda.is_available():
            used, total = torch.cuda.mem_get_info()
            ratio = 1 - used / total
            pct = int(100 * (1 - ratio))
            self.gpu_bar["value"] = pct
            self.lbl_gpu.config(text=f"GPU: {pct}% usado")
        else:
            self.lbl_gpu.config(text="GPU: CPU mode")
            self.gpu_bar["value"] = 0

        self.root.after(self.REFRESH_MS, self._tick)

    # ---------------------------------------------------------------------
    # Execução
    # ---------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gui = VoiceTypingGUI()
    gui.run()
