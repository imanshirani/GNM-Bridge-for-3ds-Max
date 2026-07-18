"""
Launch script for GNM Head Generator inside 3ds Max.

Usage in Max Python listener or MAXScript:
    import sys
    sys.path.insert(0, r"G:/3ds max Rnd/python")
    import GNM.launch
    GNM.launch.launch()
"""

import sys
import importlib
from pathlib import Path

# Add parent directory to sys.path so the GNM package is importable
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

_dock = None
_widget = None
_time_cb_tool = None   # GNMTool instance when animation callback is armed


def _evict_gnm_modules():
    """Remove all cached GNM modules so the next import re-executes them fresh.

    This is required after a pip install so that module-level flags like
    _NUMPY_OK get re-evaluated against the newly installed packages.
    """
    to_delete = [k for k in sys.modules if k == "GNM" or k.startswith("GNM.")]
    for key in to_delete:
        del sys.modules[key]
    importlib.invalidate_caches()


def _register_time_callback(tool_instance):
    """Arm the GNM animation scrub callback. Called from the Animation tab."""
    global _time_cb_tool
    _time_cb_tool = tool_instance
    try:
        import pymxs
        pymxs.runtime.execute(
            'fn gnm_time_cb = '
            '( python.execute "import GNM.launch as _gl; _gl._on_time_change()" )\n'
            'registerTimeCallback gnm_time_cb'
        )
    except Exception as e:
        print(f"[GNM] registerTimeCallback failed: {e}")


def _unregister_time_callback():
    """Disarm the GNM animation scrub callback."""
    global _time_cb_tool
    _time_cb_tool = None
    try:
        import pymxs
        pymxs.runtime.execute('unregisterTimeCallback gnm_time_cb')
    except Exception:
        pass


def _on_time_change():
    """Called by Max every time sliderTime changes. Runs on Max's main thread."""
    if _time_cb_tool is None:
        return
    try:
        import pymxs
        frame = int(pymxs.runtime.currentTime.frame)
        _time_cb_tool.apply_params_at_frame(frame)
    except Exception as e:
        print(f"[GNM] time callback error: {e}")


def _close_existing():
    """Close and fully destroy any previously opened dock / widget."""
    global _dock, _widget
    _unregister_time_callback()
    from PySide6 import QtWidgets

    if _dock is not None:
        try:
            _dock.close()
            _dock.setWidget(None)
            _dock.deleteLater()
        except Exception:
            pass
        _dock = None

    if _widget is not None:
        try:
            _widget.close()
            _widget.deleteLater()
        except Exception:
            pass
        _widget = None

    # Let Qt process pending close/destroy events before we create new widgets
    QtWidgets.QApplication.processEvents()


def _get_max_main_window():
    try:
        import qtmax
        return qtmax.GetQMaxMainWindow()
    except Exception:
        pass
    try:
        import pymxs
        from PySide6 import QtWidgets
        hwnd = pymxs.runtime.windows.getMAXHWND()
        win = QtWidgets.QWidget.find(hwnd)
        if win:
            return win
    except Exception:
        pass
    return None


def launch():
    """Open (or reopen) the GNM Head Generator panel."""
    global _dock, _widget

    _close_existing()

    # Fresh import of all GNM modules so _NUMPY_OK etc. are re-evaluated
    _evict_gnm_modules()

    try:
        from GNM.core import GNMTool
        from GNM.constants import TOOL_NAME
        from PySide6 import QtWidgets, QtCore
    except Exception as e:
        print(f"[GNM] Import error: {e}")
        import traceback
        traceback.print_exc()
        return None

    main_win = _get_max_main_window()

    try:
        _widget = GNMTool(parent=main_win)
    except Exception as e:
        print(f"[GNM] Failed to create GNMTool: {e}")
        import traceback
        traceback.print_exc()
        return None

    if main_win is not None:
        _dock = QtWidgets.QDockWidget(TOOL_NAME, main_win)
        _dock.setWidget(_widget)
        _dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        _dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        main_win.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, _dock)
        _dock.show()
    else:
        _widget.setWindowFlags(QtCore.Qt.WindowType.Window)
        _widget.show()

    return _widget


if __name__ == "__main__":
    launch()
