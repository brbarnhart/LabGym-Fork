"""Provide functions to obtain, store, and forward registration info.

"Public" Functions
	is_registered() -> bool
		Return True if registration data is stored.

	register() -> None
		Get reg info from user, store reginfo locally, and send to receiver.

	get_reginfo_from_file() -> dict | None
		Load registration info from file, and return reginfo.

Example 1
	import registration

	if not registration.is_registered():
		# Get reg info from user, store reginfo locally.  Also, send
		# reginfo to central receiver via central_logger (unless
		# central_logger's disabled attribute is True).
		registration.register()

Example 2
	import registration

	reginfo = registration.get_reginfo_from_file()

	# if user has skipped registration (with checked "Don't ask again")
	# but that was selected in some different (earlier?) installation,
	# then expire or void the "skip-henceforth" behavior.
	skip_pass_void = (reginfo is not None
		and reginfo.get('name') == 'skip'
		and packaging.version.parse(version)
			!= packaging.version.parse(reginfo.get('version'))
		)

	# if not registration.is_registered():
	if not registration.is_registered() or skip_pass_void:
		# Get reg info from user, store reginfo locally.  Also, send
		# reginfo to central receiver via central_logger (unless
		# central_logger's disabled attribute is True).
		registration.register()

Strengths
	*   Input text fields are validated as not-empty before acceptance.
	*   On macOS, displays form on foreground/top, instead of hidden
		below other windows.

Weaknesses
	*   This module file is long.  It should be refactored into a
		package with smaller module files.

"Private" Functions
	These functions are implementation details and should not be relied
	upon by external code, as they might change without notice in future
	versions.
	By convention they are named with a single leading underscore ("_")
	to indicate to other programmers that they are intended for private
	or internal use.

	_get_reginfo_from_form() -> dict | None
		Display a reg form, get user input, and return reginfo.

	_store_reginfo_to_file(reginfo: dict) -> None
		Store registration info to file in user's LabGym config directory.
"""

# Allow use of newer syntax Python 3.10 type hints in Python 3.9.
from __future__ import annotations

# Standard library imports.
from datetime import datetime
import getpass
import logging
from pathlib import Path
import platform
import sys
import textwrap
import uuid
from zoneinfo import ZoneInfo

# Related third party imports.
import yaml  # PyYAML, YAML parser and emitter for Python

# Local application/library specific imports.
from LabGym import __version__
from LabGym import central_logging
from LabGym import config


logger = logging.getLogger(__name__)


def _ensure_qt_app():
	"""Return a QApplication instance, creating one if needed."""
	from PySide6.QtWidgets import QApplication

	app = QApplication.instance()
	if app is None:
		app = QApplication(sys.argv if hasattr(sys, "argv") else [])
	return app


class RegFormDialog:
	"""Qt registration dialog (replaces the former wx form)."""

	def __init__(self, parent=None):
		from PySide6.QtCore import Qt
		from PySide6.QtGui import QFont
		from PySide6.QtWidgets import (
			QCheckBox,
			QDialog,
			QDialogButtonBox,
			QFormLayout,
			QLabel,
			QLineEdit,
			QMessageBox,
			QVBoxLayout,
		)

		title = "LabGym User Group Registration"
		header = textwrap.dedent(
			"""
			Please register to be enrolled in the LabGym User Group.

			The LabGym User Group promotes engagement between new users,
			experienced users, and developers, leading to improvements
			in user experience, including better features, better
			implementation, and better installation.
			"""
		).strip()

		self._dialog = QDialog(parent)
		self._dialog.setWindowTitle(title)
		self._dialog.setModal(True)
		self._dialog.setMinimumWidth(420)

		layout = QVBoxLayout(self._dialog)
		header_lbl = QLabel(header)
		header_lbl.setWordWrap(True)
		layout.addWidget(header_lbl)

		form = QFormLayout()
		self.input_name = QLineEdit()
		self.input_affiliation = QLineEdit()
		self.input_email = QLineEdit()
		form.addRow("Name:", self.input_name)
		form.addRow("Affiliation:", self.input_affiliation)
		form.addRow("Email Address:", self.input_email)
		layout.addLayout(form)

		self.my_checkbox = QCheckBox("Remember my choice, don't ask me again")
		layout.addWidget(self.my_checkbox)

		buttons = QDialogButtonBox()
		self.register_button = buttons.addButton(
			"Register", QDialogButtonBox.ButtonRole.AcceptRole
		)
		self.skip_button = buttons.addButton(
			"Skip for now", QDialogButtonBox.ButtonRole.RejectRole
		)
		font = self.register_button.font()
		font.setBold(True)
		font.setPointSize(font.pointSize() + 2)
		self.register_button.setFont(font)
		layout.addWidget(buttons)

		self._accepted = False

		def on_register() -> None:
			name = self.input_name.text().strip()
			affiliation = self.input_affiliation.text().strip()
			email = self.input_email.text().strip()
			if not name or not affiliation or not email:
				QMessageBox.warning(
					self._dialog,
					"Error",
					"Name, affiliation, and email cannot be empty.",
				)
				return
			self._accepted = True
			self._dialog.accept()

		def on_skip() -> None:
			self._accepted = False
			self._dialog.reject()

		self.register_button.clicked.connect(on_register)
		self.skip_button.clicked.connect(on_skip)

	def exec(self) -> bool:
		"""Show modal dialog; return True if Register was accepted."""
		self._dialog.exec()
		return self._accepted

	def GetInputValues(self, alt=None) -> dict | None:
		"""Return a dict containing the dialog object's input values."""
		if alt is None:
			return {
				"name": self.input_name.text().strip(),
				"affiliation": self.input_affiliation.text().strip(),
				"email": self.input_email.text().strip(),
			}
		if alt == "skip":
			return {
				"name": "skip",
				"affiliation": "skip",
				"email": "skip",
			}
		logger.warning("%s: %r", "Unexpected!  alt", alt)
		return {}


def _get_reginfo_from_form() -> dict | None:
	"""Display a Qt reg form, get user input, and return reginfo."""
	_ensure_qt_app()
	dlg = RegFormDialog(None)
	logger.debug("%s -- %s", "Milestone ShowModal", "calling...")
	if dlg.exec():
		logger.debug("%s -- %s", "Milestone ShowModal", "returned")
		logger.debug("User pressed [Register]")
		reginfo = dlg.GetInputValues()
	else:
		logger.debug("%s -- %s", "Milestone ShowModal", "returned")
		logger.debug("User pressed [Skip]")
		if dlg.my_checkbox.isChecked():
			logger.debug("Checked")
			reginfo = dlg.GetInputValues("skip")
		else:
			logger.debug("Unchecked")
			reginfo = None

	logger.debug("%s: %r", "reginfo", reginfo)
	return reginfo


def register(central_logger=None) -> None:
	"""Get reg info from user, store reginfo locally, and send to receiver.

	1.  Get reg info from user.
	2.  Add info from a survey of context.
	3.  Store reginfo locally.
	4.  Send reginfo to central receiver via central_logger (unless
		central_logger's disabled attribute is True).

	In production use, central_logger is not passed in, central_logger is
	obtained by calling central_logging.get_central_logger.

	For development and testing, central_logger may be overridden by the
	caller, like
		registration.register(central_logger=logging.getLogger('Local Logger'))

	(I'm ambivalent on this... central_logger could be made a required arg,
	and then no need to obtain it independently from inside this function.)
	"""
	if central_logger is None:
		central_logger = central_logging.get_central_logger()

	# pylint: disable-next=redefined-outer-name
	reginfo = _get_reginfo_from_form()
	logger.debug('%s: %r', 'reginfo', reginfo)

	if reginfo is None:
		return

	# update reginfo dict with supplemental info from a survey of context
	reginfo.update({
		'schema': 'reginfo 2025-07-10',
		'username': getpass.getuser(),

		'datetime': datetime.now(ZoneInfo('US/Eastern')).strftime(
			'%Y-%m-%dT%H:%M:%S%z'),
		'uuid': str(uuid.uuid4()),

		'platform': platform.platform(),
		'node': platform.node(),
		'version': __version__,  # LabGym version
		})

	try:
		_store_reginfo_to_file(reginfo)
		reginfo.update({'status': 'saved to regfile'})
	except Exception as e:
		reginfo.update({'status': 'unable to save to regfile'})

	central_logger.info(reginfo)


# pylint: disable-next=redefined-outer-name
def _store_reginfo_to_file(reginfo: dict) -> None:
	"""Store registration info to file in user's LabGym config directory.

	Save/store/stow/write reginfo dict to ~/.labgym/registration.yaml.
	Or more accurately, <configdir>/registration.yaml.

	Notes to developer
	*   Consider pros/cons of saving in a more opaque form.
		zip-file instead of yaml?  with '.done' extension instead of
		'.zip' so it looks like a flag-file instead of a discardable
		backup.
	*   Re naming the reciprocal functions, write/read?, store/recall?
		backup, dump, put, save, store, stow, write
		get, load, read, recall, restore
	"""

	# Get all of the values needed from config.get_config().
	configdir: Path = config.get_config()['configdir']

	# ensure configdir exists
	configdir.mkdir(parents=True, exist_ok=True)

	# write reginfo file
	regfile = configdir.joinpath('registration.yaml')
	with open(regfile, 'w', encoding='utf-8') as f:
		yaml.dump(reginfo, f, default_flow_style=False)


def get_reginfo_from_file() -> dict | None:
	"""Load registration info from file, and return reginfo."""

	# Get all of the values needed from config.get_config().
	configdir: Path = config.get_config()['configdir']

	# read reginfo file
	regfile = configdir.joinpath('registration.yaml')

	try:
		with open(regfile, 'r', encoding='utf-8') as f:
			result = yaml.safe_load(f)
	except Exception:
		result = None

	return result


def is_registered() -> bool:
	"""Return True if there is a readable regfile."""
	return not get_reginfo_from_file() is None


if __name__ == '__main__':  # pragma: no cover
	logging.basicConfig(level=logging.DEBUG,
		datefmt='%H:%M:%S',
		format='%(asctime)s\t%(levelname)s\t%(module)s:%(lineno)d\t%(message)s'
		)

	reginfo = _get_reginfo_from_form()
	logger.debug('%s: %r', 'reginfo', reginfo)
