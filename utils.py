from PySide6.QtCore import QObject, Signal


class QLogger(QObject):
    sig_log = Signal(str, str)

    def info(self, msg):
        self.sig_log.emit(f"[INFO]  {msg}", "#00FF00")

    def warning(self, msg):
        self.sig_log.emit(f"[WARN]  {msg}", "#FFA500")

    def error(self, msg):
        self.sig_log.emit(f"[ERROR] {msg}", "#FF4444")

    def section(self, msg):
        self.sig_log.emit(f"{'─' * 40}", "#555555")
        self.sig_log.emit(f"  {msg}", "#00AAFF")
        self.sig_log.emit(f"{'─' * 40}", "#555555")
