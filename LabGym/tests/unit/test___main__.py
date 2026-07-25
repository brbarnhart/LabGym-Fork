import importlib
import logging
import sys

from packaging import version

import pytest


logger = logging.getLogger(__name__)


def test_main_default_workbench(monkeypatch):
	"""Default LabGym entry launches the PySide workbench, not wx."""
	from LabGym import mylogging
	monkeypatch.setattr(mylogging, 'configure', lambda *args: None)

	from LabGym import __main__
	monkeypatch.setattr(__main__.probes, 'probes', lambda: None)
	monkeypatch.setattr(__main__, '_maybe_print_upgrade_hint', lambda: None)

	called = {'workbench': False, 'legacy': False}

	def fake_workbench():
		called['workbench'] = True

	def fake_legacy():
		called['legacy'] = True

	monkeypatch.setattr(__main__, '_main_workbench', fake_workbench)
	monkeypatch.setattr(__main__, '_main_legacy_wx', fake_legacy)
	monkeypatch.setattr(
		__main__.config,
		'get_config',
		lambda: {'selftest': False, 'legacy_wx': False},
	)

	__main__.main()
	assert called['workbench'] is True
	assert called['legacy'] is False


def test_main_legacy_wx_flag(monkeypatch):
	"""``--legacy-wx`` routes to the classic wx GUI helper."""
	from LabGym import mylogging
	monkeypatch.setattr(mylogging, 'configure', lambda *args: None)

	from LabGym import __main__
	monkeypatch.setattr(__main__, '_maybe_print_upgrade_hint', lambda: None)

	called = {'workbench': False, 'legacy': False}

	monkeypatch.setattr(
		__main__,
		'_main_workbench',
		lambda: called.__setitem__('workbench', True),
	)
	monkeypatch.setattr(
		__main__,
		'_main_legacy_wx',
		lambda: called.__setitem__('legacy', True),
	)
	monkeypatch.setattr(
		__main__.config,
		'get_config',
		lambda: {'selftest': False, 'legacy_wx': True},
	)

	__main__.main()
	assert called['legacy'] is True
	assert called['workbench'] is False


def test_main_current_labgym(monkeypatch):
	# Arrange
	from LabGym import mylogging
	monkeypatch.setattr(mylogging, 'configure', lambda *args: None)

	from LabGym import __main__
	return_values = [
		version.Version('2.8.16'),
		version.Version('2.8.16'),
	]
	monkeypatch.setattr(
		__main__.version, 'parse', lambda self: return_values.pop(0)
	)

	monkeypatch.setattr(__main__.probes, 'probes', lambda: None)
	monkeypatch.setattr(__main__, '_main_workbench', lambda: None)
	monkeypatch.setattr(__main__, '_main_legacy_wx', lambda: None)
	monkeypatch.setattr(
		__main__.config,
		'get_config',
		lambda: {'selftest': False, 'legacy_wx': False},
	)

	# Act
	__main__.main()


def test_parse_args_legacy_wx(monkeypatch):
	from LabGym import myargparse

	monkeypatch.setattr(sys, 'argv', ['cmd', '--legacy-wx'])
	result = myargparse.parse_args()
	assert result.get('legacy_wx') is True


def test_legacy_launch_uses_legacy_flag(monkeypatch):
	import LabGym.gui_pyside.legacy_launch as mod

	captured = {}

	class FakePopen:
		def __init__(self, cmd, **kwargs):
			captured['cmd'] = list(cmd)

	monkeypatch.setattr(mod.subprocess, 'Popen', FakePopen)
	mod.launch_legacy_labgym()
	assert '-m' in captured['cmd']
	assert 'LabGym' in captured['cmd']
	assert '--legacy-wx' in captured['cmd']