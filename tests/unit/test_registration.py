import logging
from pathlib import Path

import pytest

from LabGym import registration


testdir = Path(__file__[:-3])  # dir containing support files for unit tests
assert testdir.is_dir()


def test_dummy():
	pass


def test_get_reginfo_from_file(monkeypatch):
	_config = {
		'configdir': testdir,
	}
	monkeypatch.setattr(registration.config, 'get_config', lambda: _config)

	result = registration.get_reginfo_from_file()
	assert result.get('schema') == 'reginfo 2025-07-10'


def test_is_registered(monkeypatch, tmp_path):
	_config = {
		'configdir': tmp_path,
	}
	monkeypatch.setattr(registration.config, 'get_config', lambda: _config)

	result = registration.is_registered()
	assert result is False


def test_register_skip(monkeypatch, tmp_path):
	"""register() with a mocked form that skips without 'remember' stores nothing."""
	_config = {'configdir': tmp_path}
	monkeypatch.setattr(registration.config, 'get_config', lambda: _config)
	monkeypatch.setattr(registration, '_get_reginfo_from_form', lambda: None)

	registration.register(logging.getLogger('test'))
	assert registration.is_registered() is False


def test_register_skip_remember(monkeypatch, tmp_path):
	"""Skip with remember writes a skip regfile."""
	_config = {'configdir': tmp_path}
	monkeypatch.setattr(registration.config, 'get_config', lambda: _config)
	monkeypatch.setattr(
		registration,
		'_get_reginfo_from_form',
		lambda: {'name': 'skip', 'affiliation': 'skip', 'email': 'skip'},
	)

	registration.register(logging.getLogger('test'))
	assert registration.is_registered() is True
	info = registration.get_reginfo_from_file()
	assert info.get('name') == 'skip'


logging.getLogger().setLevel(logging.DEBUG)
