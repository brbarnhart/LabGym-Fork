'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext.

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''


# noqa

# Standard library imports.
import logging
from pathlib import Path
import sys

# Regarding the use of logging and mylogging functions before the
# "Related third party imports." section:
# These statements are intentionally positioned before this module's
# other imports (against the guidance of PEP 8), to log the loading of
# this module before other import statements are executed and
# potentially produce their own log messages.
from LabGym import mylogging
# Collect logrecords and defer handling until logging is configured.
mylogging.defer()

# Log the load of this module (by the module loader, on first import).
logger = logging.getLogger(__name__)
logger.debug('%s', f'loading {__name__}')

# Configure logging based on configfile, then handle collected logrecords.
mylogging.configure()

# Related third party imports.
from packaging import version  # Core utilities for Python packages
import requests  # Python HTTP for Humans.

# Local application/library specific imports.
# pylint: disable=ungrouped-imports
# pylint: disable-next=unused-import
from LabGym import mypkg_resources  # replace deprecated pkg_resources
from LabGym import __version__, probes
from LabGym import config, selftest


logger.debug('%s: %r', '(__name__, __package__)', (__name__, __package__))


def _maybe_print_upgrade_hint() -> None:
	"""Print a non-fatal upgrade notice when PyPI reports a newer version."""
	try:
		current_version = version.parse(__version__)
		logger.debug('%s: %r', 'current_version', current_version)
		pypi_json = requests.get('https://pypi.org/pypi/LabGym/json', timeout=5).json()
		latest_version = version.parse(pypi_json['info']['version'])
		logger.debug('%s: %r', 'latest_version', latest_version)

		if latest_version > current_version:
			if 'pipx' in str(Path(__file__)):
				upgrade_command = 'pipx upgrade LabGym'
			else:
				upgrade_command = 'python3 -m pip install --upgrade LabGym'

			print(
				f'You are using LabGym {current_version}, but version '
				f'{latest_version} is available.'
			)
			print(f'Consider upgrading LabGym by using the command "{upgrade_command}".')
			print(
				'For the details of new changes, check '
				'https://github.com/umyelab/LabGym.\n'
			)
	except Exception:
		pass


def _main_workbench() -> None:
	"""Default UI: PySide6 FreeCAD-style workbench shell (Phase 8)."""
	logger.info('Starting LabGym workbench shell (PySide6)')
	# Registration / userdata probes do not require wx.
	probes.probes()
	from LabGym.gui_pyside.main_window import main as workbench_main

	workbench_main()


def _main_legacy_wx() -> None:
	"""Deprecated classic wxPython GUI (``LabGym --legacy-wx``)."""
	logger.warning(
		'Starting legacy wxPython LabGym GUI (--legacy-wx). '
		'This interface is deprecated; use the default workbench shell when possible.'
	)
	print(
		'NOTE: The classic wxPython LabGym window is deprecated.\n'
		'      Prefer ``LabGym`` (PySide workbench). Use ``LabGym --legacy-wx`` only if needed.\n'
	)

	# On Windows, PyTorch must be imported before wxPython. Loading wx first can
	# leave DLLs in a state that makes torch fail with WinError 1114 on c10.dll.
	import torch  # noqa: F401  # pylint: disable=unused-import

	from LabGym import mywx  # on load, monkeypatch wx.App to be a strict-singleton
	import wx  # wxPython, Cross platform GUI toolkit for Python, "Phoenix" version
	from LabGym import gui_main

	# Create a single persistent, wx.App instance, as it may be
	# needed for probe dialogs prior to calling gui_main.main_window.
	assert wx.GetApp() is None
	wx.App()
	mywx.bring_wxapp_to_foreground()

	# Perform some pre-op sanity checks and probes of outside resources.
	probes.probes()

	gui_main.main_window()


def main() -> None:
	"""Launch LabGym: workbench by default, or legacy wx with ``--legacy-wx``."""

	# Get all of the values needed from config.get_config().
	cfg = config.get_config()
	flag_selftest: bool = bool(cfg.get('selftest', False))
	flag_legacy_wx: bool = bool(cfg.get('legacy_wx', False))

	if flag_selftest:
		logger.info('%s -- %s', 'run_selftests()', 'calling...')
		result = selftest.run_selftests()
		logger.info('%s -- %s', 'run_selftests()', f'returned {result!r}')
		logger.info('%s -- %s', f'sys.exit({result!r})', 'calling...')
		sys.exit(result)

	_maybe_print_upgrade_hint()

	if flag_legacy_wx:
		_main_legacy_wx()
	else:
		_main_workbench()

	logger.debug('Milestone -- exiting main')


if __name__ == '__main__':  # pragma: no cover

	main()
