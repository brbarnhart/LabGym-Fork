import logging
import os
from pathlib import Path
import textwrap

import pytest  # pytest: simple powerful testing with Python

from LabGym import userdata_survey
from .exitstatus import exitstatus


@pytest.fixture
def silence_ok_dialog(monkeypatch):
	"""Avoid opening real Qt dialogs in unit tests."""
	monkeypatch.setattr(userdata_survey, "_ok_dialog", lambda title, msg: None)


def test_dummy():
	pass


def test_is_path_under():
	assert userdata_survey.is_path_under('/a/b', '/a/b') == False
	assert userdata_survey.is_path_under('/a/b', '/a/b/..') == False
	assert userdata_survey.is_path_under('/a/b', '/a/b/c') == True
	assert userdata_survey.is_path_under('/a/b', '/a/b/c/d') == True
	assert userdata_survey.is_path_under('/a/b', '/a/c/../b/d') == True
	assert userdata_survey.is_path_under('/a/b/c', '/a/b') == False
	assert userdata_survey.is_path_under('/a/b', '/a/c') == False
	assert userdata_survey.is_path_under('/a/b', '/a/b/../c') == False


def test_is_path_equivalent():
	assert userdata_survey.is_path_equivalent('/a/b', '/a/b') == True
	assert userdata_survey.is_path_equivalent('/a/b', '/a/c') == False


def test_dict2str():
	myarg = {'a': 'A', 'c': 'C', 'b': 'B'}
	hanging_indent = ' ' * 16
	expected = textwrap.dedent(f"""\
		a: A
		{hanging_indent}c: C
		{hanging_indent}b: B
		""").strip()
	result = userdata_survey.dict2str(myarg)
	assert result == expected

	myarg = {'a': 'A', 'c': 'C', 'b': 'B'}
	hanging_indent = ' ' * 2
	expected = textwrap.dedent(f"""\
		a: A
		{hanging_indent}c: C
		{hanging_indent}b: B
		""").strip()
	result = userdata_survey.dict2str(myarg, hanging_indent=hanging_indent)
	assert result == expected


def test_get_list_of_subdirs(tmp_path):
	(Path(tmp_path) / 'alfa').mkdir()
	(Path(tmp_path) / 'bravo').mkdir()
	(Path(tmp_path) / 'charlie').mkdir()
	(Path(tmp_path) / '__pycache__').mkdir()
	(Path(tmp_path) / '__init__.py').touch()

	expected = ['alfa', 'bravo', 'charlie']
	result = userdata_survey.get_list_of_subdirs(tmp_path)
	assert result == expected


def test_assert_userdata_dirs_are_separate(monkeypatch, tmp_path, silence_ok_dialog, caplog):
	with pytest.raises(SystemExit, match="Bad configuration") as e:
		userdata_survey.assert_userdata_dirs_are_separate(tmp_path, tmp_path)

	assert exitstatus(e.value) == 1
	expected_msg = textwrap.dedent("""
		LabGym Configuration Error
		The userdata folders must be separate.
		""").strip()
	assert expected_msg in caplog.text


def test_survey_case1(monkeypatch, tmp_path, silence_ok_dialog, caplog):
	monkeypatch.setattr(
		userdata_survey.config,
		'get_config',
		lambda: {'enable': {'assess_userdata_folders': True}},
	)
	labgym = os.path.join(tmp_path, 'LabGym')
	detectors = os.path.join(tmp_path, 'detectors')
	models = os.path.join(tmp_path, 'detectors', 'models')

	with pytest.raises(SystemExit, match="Bad configuration") as e:
		userdata_survey.survey(labgym, detectors, models)

	assert exitstatus(e.value) == 1
	expected_msg = textwrap.dedent("""
		LabGym Configuration Error
		The userdata folders must be separate.
		""").strip()
	assert expected_msg in caplog.text


def test_survey_case2(monkeypatch, tmp_path, silence_ok_dialog, caplog):
	monkeypatch.setattr(
		userdata_survey.config,
		'get_config',
		lambda: {'enable': {'assess_userdata_folders': True}},
	)
	labgym = os.path.join(tmp_path, 'LabGym')
	detectors = os.path.join(tmp_path, 'detectors')
	models = os.path.join(tmp_path, 'models')

	userdata_survey.survey(labgym, detectors, models)

	expected_msg = "External Userdata folders specified by config don't exist."
	assert expected_msg in caplog.text


def test_survey_case3(monkeypatch, tmp_path, silence_ok_dialog, caplog):
	monkeypatch.setattr(
		userdata_survey.config,
		'get_config',
		lambda: {'enable': {'assess_userdata_folders': True}},
	)
	labgym = os.path.join(tmp_path, 'LabGym')
	detectors = os.path.join(tmp_path, 'LabGym', 'detectors')
	models = os.path.join(tmp_path, 'LabGym', 'models')

	userdata_survey.survey(labgym, detectors, models)

	expected_msg = "Found internal Userdata folders specified by config."
	assert expected_msg in caplog.text


def test_survey_case4(monkeypatch, tmp_path, silence_ok_dialog, caplog):
	monkeypatch.setattr(
		userdata_survey.config,
		'get_config',
		lambda: {'enable': {'assess_userdata_folders': True}},
	)
	labgym = os.path.join(tmp_path, 'LabGym')
	detectors = os.path.join(tmp_path, 'detectors')
	models = os.path.join(tmp_path, 'models')
	os.mkdir(detectors)
	os.mkdir(models)

	orphans = [
		os.path.join(labgym, 'detectors', 'detector1'),
		os.path.join(labgym, 'detectors', 'detector2'),
		os.path.join(labgym, 'detectors', '__pycache__'),
		os.path.join(labgym, 'models', 'model1'),
		os.path.join(labgym, 'models', 'model2'),
		os.path.join(labgym, 'models', '__pycache__'),
	]
	for orphan in orphans:
		os.makedirs(orphan)

	(Path(labgym) / 'detectors' / '__init__.py').touch()
	(Path(labgym) / 'models' / '__init__.py').touch()

	userdata_survey.survey(labgym, detectors, models)

	assert "orphan" in caplog.text.lower() or "internal" in caplog.text.lower()
