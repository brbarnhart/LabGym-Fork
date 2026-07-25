import importlib
import logging
import sys

from packaging import version

import pytest


logger = logging.getLogger(__name__)


def test_main_default_workbench(monkeypatch):
	"""Default LabGym entry launches the PySide workbench."""
	from LabGym import mylogging
	monkeypatch.setattr(mylogging, 'configure', lambda *args: None)

	from LabGym import __main__
	monkeypatch.setattr(__main__.probes, 'probes', lambda: None)
	monkeypatch.setattr(__main__, '_maybe_print_upgrade_hint', lambda: None)

	called = {'workbench': False}

	def fake_workbench():
		called['workbench'] = True

	monkeypatch.setattr(__main__, '_main_workbench', fake_workbench)
	monkeypatch.setattr(
		__main__.config,
		'get_config',
		lambda: {'selftest': False},
	)

	__main__.main()
	assert called['workbench'] is True


def test_main_current_labgym(monkeypatch):
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
	monkeypatch.setattr(
		__main__.config,
		'get_config',
		lambda: {'selftest': False},
	)

	__main__.main()
