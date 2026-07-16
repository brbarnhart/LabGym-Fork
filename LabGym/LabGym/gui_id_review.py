'''Full-video timeline UI for ID switch annotation.'''

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import wx

from .id_review.dataset import (
	finalize_switch_annotations,
	load_events,
	load_switches,
	make_swap_marker,
)
from .id_review.types import ContactEvent, ReviewDecision, SwitchMarker, TrackletStore
from .id_review.tracklets import load_tracklets
from .id_review.samples import (
	analysis_frame_to_video_frame,
	detections_at_frame_after_markers,
	draw_detections_overlay,
)


class RiskTimeline(wx.Panel):
	'''Paint risk bands, switch markers, and playhead; click to seek.'''

	def __init__(self, parent, on_seek=None, height: int = 56):
		super().__init__(parent, size=(-1, height))
		self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
		self.n_frames = 1
		self.frame = 0
		self.events: List[ContactEvent] = []
		self.markers: List[SwitchMarker] = []
		self.min_risk = 0.0
		self.on_seek = on_seek
		self.Bind(wx.EVT_PAINT, self._on_paint)
		self.Bind(wx.EVT_LEFT_DOWN, self._on_click)
		self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))

	def set_data(
		self,
		n_frames: int,
		frame: int,
		events: Sequence[ContactEvent],
		markers: Sequence[SwitchMarker],
		min_risk: float = 0.0,
	):
		self.n_frames = max(1, int(n_frames))
		self.frame = int(np.clip(frame, 0, self.n_frames - 1))
		self.events = list(events)
		self.markers = list(markers)
		self.min_risk = float(min_risk)
		self.Refresh()

	def _x_to_frame(self, x: int) -> int:
		w = max(1, self.GetClientSize().width)
		return int(np.clip(round(x / w * (self.n_frames - 1)), 0, self.n_frames - 1))

	def _frame_to_x(self, f: int) -> int:
		w = max(1, self.GetClientSize().width)
		if self.n_frames <= 1:
			return 0
		return int(round(f / (self.n_frames - 1) * (w - 1)))

	def _on_click(self, event):
		f = self._x_to_frame(event.GetX())
		if self.on_seek:
			self.on_seek(f)

	def _on_paint(self, event):
		dc = wx.AutoBufferedPaintDC(self)
		w, h = self.GetClientSize()
		dc.SetBackground(wx.Brush(wx.Colour(30, 30, 34)))
		dc.Clear()

		# baseline
		dc.SetPen(wx.Pen(wx.Colour(80, 80, 90), 1))
		dc.DrawLine(0, h // 2, w, h // 2)

		# risk bands
		for ev in self.events:
			if ev.risk_score < self.min_risk:
				continue
			x0 = self._frame_to_x(ev.start_frame)
			x1 = max(x0 + 2, self._frame_to_x(ev.end_frame))
			# orange -> red by risk
			r = int(180 + 75 * min(1.0, ev.risk_score))
			g = int(120 * (1.0 - min(1.0, ev.risk_score)))
			b = 40
			if 'possible_swap' in (ev.risk_flags or []):
				g = min(g, 60)
				r = 255
			dc.SetBrush(wx.Brush(wx.Colour(r, g, b, 180)))
			dc.SetPen(wx.Pen(wx.Colour(r, g, b), 1))
			dc.DrawRectangle(x0, 8, max(2, x1 - x0), h - 16)

		# switch markers
		dc.SetPen(wx.Pen(wx.Colour(80, 255, 120), 2))
		for m in self.markers:
			x = self._frame_to_x(m.frame)
			dc.DrawLine(x, 2, x, h - 2)
			dc.SetBrush(wx.Brush(wx.Colour(80, 255, 120)))
			dc.DrawCircle(x, 10, 4)

		# playhead
		px = self._frame_to_x(self.frame)
		dc.SetPen(wx.Pen(wx.Colour(255, 255, 0), 2))
		dc.DrawLine(px, 0, px, h)
		dc.SetTextForeground(wx.Colour(200, 200, 200))
		dc.DrawText('risk bands  |  green = switch markers  |  yellow = playhead', 4, h - 16)


class FullVideoIdReviewDialog(wx.Dialog):
	'''
	Full-video ID review: timeline risk highlights + user switch timestamps.
	'''

	def __init__(
		self,
		parent,
		review_dir: str,
		events: Optional[List[ContactEvent]] = None,
	):
		super().__init__(
			parent,
			title='ID review — full video timeline',
			style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX,
		)
		self.review_dir = review_dir
		self.events = events if events is not None else load_events(review_dir)
		self.markers: List[SwitchMarker] = load_switches(review_dir)
		self._undo_stack: List[List[SwitchMarker]] = []  # snapshots before each edit
		self._stores: Dict[str, TrackletStore] = {}
		self._cap: Optional[cv2.VideoCapture] = None
		self._cap_path: Optional[str] = None
		self._play_timer = wx.Timer(self)
		self._playing = False
		self._updating = False
		self.frame = 0
		self.n_frames = 1
		self.fps = 10.0
		self.animal_kind = 'mouse'
		self.involved_ids: List[int] = [0, 1]
		self.min_risk = 0.0

		self._load_stores()
		self._init_span()

		self._build_ui()
		self.Bind(wx.EVT_TIMER, self._on_play_tick, self._play_timer)
		self.Bind(wx.EVT_CLOSE, self._on_close)
		self.Bind(wx.EVT_CHAR_HOOK, self._on_key)

		self._seek(self.frame)
		self.SetSize((1100, 820))
		if parent is not None:
			self.CentreOnParent()
		else:
			self.Centre()

	def _load_stores(self):
		# load all tracklet kinds present
		for name in os.listdir(self.review_dir):
			if name.endswith('_tracklets_meta.json'):
				kind = name[: -len('_tracklets_meta.json')]
				try:
					self._stores[kind] = load_tracklets(self.review_dir, kind)
				except Exception as exc:
					print(f'ID review: load tracklets {kind} failed: {exc}', flush=True)

	def _init_span(self):
		if self._stores:
			# prefer kind with most ids / frames
			kind = max(self._stores.keys(), key=lambda k: (len(self._stores[k].ids), self._stores[k].n_frames))
			self.animal_kind = kind
			store = self._stores[kind]
			self.n_frames = max(1, store.n_frames)
			self.involved_ids = list(store.ids)
			self.fps = float(store.meta.get('fps') or (self.events[0].fps if self.events else 10) or 10)
		elif self.events:
			self.animal_kind = self.events[0].animal_kind
			self.n_frames = max(ev.end_frame for ev in self.events) + 50
			self.fps = float(self.events[0].fps or 10)
			self.involved_ids = list(self.events[0].involved_ids)
		self.frame = 0

	def _build_ui(self):
		root = wx.BoxSizer(wx.VERTICAL)

		help_txt = wx.StaticText(
			self,
			label=(
				'Orange/red timeline bands = automatic switch-risk (contact). '
				'Green ticks = your switch markers. Scrub the full video, mark where IDs actually flip, then Done. '
				'Keys: ←/→ step, Space play/pause, S mark swap, Delete remove, Ctrl+Z / U undo last change.'
			),
		)
		help_txt.Wrap(1040)
		root.Add(help_txt, 0, wx.ALL | wx.EXPAND, 8)

		self.video_bmp = wx.StaticBitmap(self, size=(720, 480))
		root.Add(self.video_bmp, 0, wx.ALL | wx.ALIGN_CENTER, 6)

		self.status = wx.StaticText(self, label='')
		root.Add(self.status, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)

		# transport
		row = wx.BoxSizer(wx.HORIZONTAL)
		self.btn_back = wx.Button(self, label='◀ -1', size=(60, -1))
		self.btn_play = wx.Button(self, label='Play', size=(70, -1))
		self.btn_fwd = wx.Button(self, label='+1 ▶', size=(60, -1))
		self.btn_prev_risk = wx.Button(self, label='⟵ Risk', size=(80, -1))
		self.btn_next_risk = wx.Button(self, label='Risk ⟶', size=(80, -1))
		self.btn_mark = wx.Button(self, label='Mark swap here (S)', size=(140, -1))
		self.btn_remove_here = wx.Button(self, label='Remove at this frame', size=(140, -1))
		self.btn_delete = wx.Button(self, label='Delete selected', size=(120, -1))
		self.btn_undo = wx.Button(self, label='Undo (Ctrl+Z)', size=(110, -1))
		wx.Button.SetToolTip(
			self.btn_remove_here,
			'Remove the switch marker at the current playhead frame (if any).',
		)
		wx.Button.SetToolTip(
			self.btn_delete,
			'Remove the marker selected in the list below. Shortcut: Delete / Backspace.',
		)
		wx.Button.SetToolTip(
			self.btn_undo,
			'Undo the last mark or delete. Shortcut: Ctrl+Z or U.',
		)
		for b, h in (
			(self.btn_back, lambda e: self._nudge(-1)),
			(self.btn_play, self._toggle_play),
			(self.btn_fwd, lambda e: self._nudge(1)),
			(self.btn_prev_risk, lambda e: self._jump_risk(-1)),
			(self.btn_next_risk, lambda e: self._jump_risk(1)),
			(self.btn_mark, lambda e: self._mark_swap()),
			(self.btn_remove_here, lambda e: self._remove_at_current_frame()),
			(self.btn_delete, self._delete_selected_marker),
			(self.btn_undo, lambda e: self._undo()),
		):
			b.Bind(wx.EVT_BUTTON, h)
			row.Add(b, 0, wx.ALL, 3)
		root.Add(row, 0, wx.ALL | wx.ALIGN_CENTER, 4)

		self.slider = wx.Slider(self, value=0, minValue=0, maxValue=max(0, self.n_frames - 1), style=wx.SL_HORIZONTAL | wx.SL_LABELS)
		self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
		root.Add(self.slider, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 12)

		self.timeline = RiskTimeline(self, on_seek=self._seek, height=60)
		root.Add(self.timeline, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 12)

		filt = wx.BoxSizer(wx.HORIZONTAL)
		filt.Add(wx.StaticText(self, label='Min risk to show on timeline:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
		self.spin_risk = wx.SpinCtrlDouble(self, min=0.0, max=1.0, inc=0.05, initial=0.0, size=(80, -1))
		self.spin_risk.SetDigits(2)
		self.spin_risk.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_risk_filter)
		filt.Add(self.spin_risk, 0, wx.RIGHT, 12)
		filt.Add(wx.StaticText(self, label='Animal kind:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
		self.kind_choice = wx.Choice(self, choices=sorted(self._stores.keys()) or [self.animal_kind])
		if self.animal_kind in (sorted(self._stores.keys()) or [self.animal_kind]):
			self.kind_choice.SetStringSelection(self.animal_kind)
		self.kind_choice.Bind(wx.EVT_CHOICE, self._on_kind)
		filt.Add(self.kind_choice, 0)
		root.Add(filt, 0, wx.ALL, 8)

		root.Add(wx.StaticText(self, label='Switch markers (ground truth):'), 0, wx.LEFT | wx.TOP, 10)
		self.marker_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL, size=(-1, 120))
		self.marker_list.InsertColumn(0, 'ID', width=90)
		self.marker_list.InsertColumn(1, 'Frame', width=70)
		self.marker_list.InsertColumn(2, 'Time (s)', width=80)
		self.marker_list.InsertColumn(3, 'IDs', width=80)
		self.marker_list.InsertColumn(4, 'Action', width=70)
		self.marker_list.InsertColumn(5, 'Linked risk', width=100)
		self.marker_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_marker_activated)
		root.Add(self.marker_list, 1, wx.ALL | wx.EXPAND, 8)

		bottom = wx.BoxSizer(wx.HORIZONTAL)
		self.btn_done = wx.Button(self, label='Done — apply switches & continue analysis')
		self.btn_cancel = wx.Button(self, label='Cancel — skip remaps')
		self.btn_done.Bind(wx.EVT_BUTTON, self._on_done)
		self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
		bottom.Add(self.btn_cancel, 0, wx.ALL, 6)
		bottom.Add(self.btn_done, 0, wx.ALL, 6)
		root.Add(bottom, 0, wx.ALIGN_RIGHT | wx.ALL, 8)

		self.SetSizer(root)
		self._refresh_marker_list()
		self._refresh_timeline()
		self._update_undo_button()

	# ----- video / seek -----

	def _primary_store(self) -> Optional[TrackletStore]:
		return self._stores.get(self.animal_kind)

	def _video_meta(self) -> Tuple[Optional[str], dict, float, Optional[int]]:
		store = self._primary_store()
		meta = dict(store.meta) if store else {}
		if self.events and self.events[0].video:
			meta.setdefault('video', self.events[0].video)
		video = meta.get('video')
		fps = float(meta.get('fps') or self.fps or 10)
		return video, meta, fps, meta.get('framewidth')

	def _ensure_cap(self, path: str) -> bool:
		if self._cap is not None and self._cap_path == path:
			return True
		if self._cap is not None:
			self._cap.release()
		cap = cv2.VideoCapture(path)
		if not cap.isOpened():
			self._cap = None
			self._cap_path = None
			return False
		self._cap = cap
		self._cap_path = path
		return True

	def _stop_play(self):
		self._playing = False
		if self._play_timer.IsRunning():
			self._play_timer.Stop()
		self.btn_play.SetLabel('Play')

	def _seek(self, frame: int):
		self.frame = int(np.clip(frame, 0, max(0, self.n_frames - 1)))
		self._updating = True
		try:
			if self.slider.GetMax() != max(0, self.n_frames - 1):
				self.slider.SetMax(max(0, self.n_frames - 1))
			self.slider.SetValue(self.frame)
		finally:
			self._updating = False
		self._render()
		self._refresh_timeline()

	def _nudge(self, d: int):
		self._stop_play()
		self._seek(self.frame + d)

	def _on_slider(self, event):
		if self._updating:
			return
		self._stop_play()
		self._seek(self.slider.GetValue())

	def _toggle_play(self, event=None):
		if self._playing:
			self._stop_play()
			return
		self._playing = True
		self.btn_play.SetLabel('Pause')
		interval = max(20, int(round(1000.0 / max(1.0, self.fps))))
		self._play_timer.Start(interval)

	def _on_play_tick(self, event):
		if not self._playing:
			return
		if self.frame >= self.n_frames - 1:
			self._stop_play()
			return
		self._seek(self.frame + 1)

	def _render(self):
		video, meta, fps, fw = self._video_meta()
		self.fps = fps
		t = self.frame / fps if fps else 0.0
		store = self._primary_store()

		if not video or not self._ensure_cap(str(video)):
			self.status.SetLabel(
				f'Frame {self.frame}/{self.n_frames - 1}  t={t:.2f}s  |  video unavailable: {video}'
			)
			self._set_placeholder()
			return

		v_idx = analysis_frame_to_video_frame(meta, self.frame, fps)
		self._cap.set(cv2.CAP_PROP_POS_FRAMES, v_idx)
		ok, frame = self._cap.read()
		if not ok or frame is None:
			self.status.SetLabel(f'Failed to read video frame {v_idx}')
			return
		if fw is not None:
			try:
				fw = int(fw)
				h, w = frame.shape[:2]
				if w != fw and fw > 0:
					frame = cv2.resize(frame, (fw, int(h * fw / w)), interpolation=cv2.INTER_AREA)
			except Exception:
				pass

		applied = [m for m in self.markers if m.frame <= self.frame and m.animal_kind == self.animal_kind]
		if store is not None:
			dets = detections_at_frame_after_markers(store, self.frame, applied)
			frame = draw_detections_overlay(
				frame,
				dets,
				highlight_ids=self.involved_ids,
				frame_idx=self.frame,
				n_markers_applied=len(applied),
			)

		n_risk = sum(1 for e in self.events if e.start_frame <= self.frame <= e.end_frame)
		self.status.SetLabel(
			f'Analysis frame {self.frame} / {self.n_frames - 1}  |  t={t:.2f}s  |  video f={v_idx}  |  '
			f'switches marked={len(self.markers)}  |  in risk band={bool(n_risk)}  |  '
			f'preview applies {len(applied)} prior marker(s)'
		)
		self._set_bgr_image(frame)

	def _set_placeholder(self):
		img = wx.Image(720, 480)
		img.SetRGB(wx.Rect(0, 0, 720, 480), 40, 40, 40)
		self.video_bmp.SetBitmap(wx.Bitmap(img))

	def _set_bgr_image(self, arr: np.ndarray, max_w=720, max_h=480):
		rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
		h, w = rgb.shape[:2]
		scale = min(max_w / w, max_h / h, 1.0)
		nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
		rgb = np.ascontiguousarray(cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA))
		self.video_bmp.SetBitmap(wx.Bitmap(wx.Image(nw, nh, rgb.tobytes())))
		self.Layout()

	# ----- markers / risk nav -----

	def _on_risk_filter(self, event):
		self.min_risk = float(self.spin_risk.GetValue())
		self._refresh_timeline()

	def _on_kind(self, event):
		self.animal_kind = self.kind_choice.GetStringSelection()
		store = self._primary_store()
		if store:
			self.n_frames = store.n_frames
			self.involved_ids = list(store.ids)
			self.fps = float(store.meta.get('fps') or self.fps)
		self._seek(min(self.frame, self.n_frames - 1))

	def _events_for_kind(self) -> List[ContactEvent]:
		return [e for e in self.events if e.animal_kind == self.animal_kind]

	def _refresh_timeline(self):
		self.timeline.set_data(
			self.n_frames,
			self.frame,
			self._events_for_kind(),
			[m for m in self.markers if m.animal_kind == self.animal_kind],
			min_risk=self.min_risk,
		)

	def _snapshot_markers(self) -> List[SwitchMarker]:
		'''Deep-enough copy of marker list for undo.'''
		return [
			SwitchMarker.from_dict(m.to_dict())
			for m in self.markers
		]

	def _push_undo(self):
		self._undo_stack.append(self._snapshot_markers())
		# cap history
		if len(self._undo_stack) > 50:
			self._undo_stack = self._undo_stack[-50:]
		self._update_undo_button()

	def _update_undo_button(self):
		if hasattr(self, 'btn_undo'):
			self.btn_undo.Enable(bool(self._undo_stack))

	def _apply_marker_change(self):
		self.markers.sort(key=lambda x: x.frame)
		self._refresh_marker_list()
		self._refresh_timeline()
		self._render()
		self._update_undo_button()

	def _select_marker_in_list(self, marker_id: str):
		for i in range(self.marker_list.GetItemCount()):
			if self.marker_list.GetItemText(i, 0) == marker_id:
				self.marker_list.Select(i)
				self.marker_list.EnsureVisible(i)
				return

	def _refresh_marker_list(self):
		self.marker_list.DeleteAllItems()
		for i, m in enumerate(sorted(self.markers, key=lambda x: x.frame)):
			self.marker_list.InsertItem(i, m.marker_id)
			self.marker_list.SetItem(i, 1, str(m.frame))
			self.marker_list.SetItem(i, 2, f'{m.time_sec:.2f}' if m.time_sec is not None else '')
			self.marker_list.SetItem(i, 3, ','.join(str(x) for x in m.involved_ids))
			self.marker_list.SetItem(i, 4, m.action)
			self.marker_list.SetItem(i, 5, m.linked_event_id or '')

	def _mark_swap(self):
		ids = self.involved_ids
		if len(ids) != 2:
			# use first two
			if len(ids) < 2:
				wx.MessageBox('Need exactly 2 animal IDs to mark a swap.', 'Cannot mark', wx.OK | wx.ICON_WARNING)
				return
			ids = ids[:2]
		self._push_undo()
		# replace existing marker at same frame
		self.markers = [m for m in self.markers if not (m.frame == self.frame and m.animal_kind == self.animal_kind)]
		try:
			m = make_swap_marker(
				self.frame,
				self.animal_kind,
				ids,
				fps=self.fps,
				marker_id=f's{self.frame:06d}_{self.animal_kind}',
			)
		except ValueError as exc:
			# roll back empty push if mark fails after push — restore
			if self._undo_stack:
				self.markers = self._undo_stack.pop()
			wx.MessageBox(str(exc), 'Cannot mark', wx.OK | wx.ICON_WARNING)
			self._update_undo_button()
			return
		# link risk
		for ev in self._events_for_kind():
			if ev.start_frame <= m.frame <= ev.end_frame + 5:
				m.linked_event_id = ev.event_id
				break
		self.markers.append(m)
		self._apply_marker_change()
		self._select_marker_in_list(m.marker_id)
		self.status.SetLabel(
			self.status.GetLabel() + f'  |  Added switch at frame {m.frame} (Undo available)'
		)

	def _delete_selected_marker(self, event=None):
		i = self.marker_list.GetFirstSelected()
		if i < 0:
			# fallback: remove at current frame if present
			if self._remove_at_current_frame(silent_if_none=True):
				return
			wx.MessageBox(
				'Select a marker in the list below, or move the playhead to a marked frame '
				'and use “Remove at this frame”.\n\n'
				'Shortcuts: Delete / Backspace, Ctrl+Z to undo.',
				'No marker selected',
				wx.OK | wx.ICON_INFORMATION,
			)
			return
		mid = self.marker_list.GetItemText(i, 0)
		self._push_undo()
		self.markers = [m for m in self.markers if m.marker_id != mid]
		self._apply_marker_change()

	def _remove_at_current_frame(self, event=None, silent_if_none: bool = False) -> bool:
		'''Remove marker(s) at the current playhead for this animal kind.'''
		to_remove = [
			m for m in self.markers
			if m.frame == self.frame and m.animal_kind == self.animal_kind
		]
		if not to_remove:
			if not silent_if_none:
				wx.MessageBox(
					f'No switch marker at frame {self.frame}.',
					'Nothing to remove',
					wx.OK | wx.ICON_INFORMATION,
				)
			return False
		self._push_undo()
		ids = {m.marker_id for m in to_remove}
		self.markers = [m for m in self.markers if m.marker_id not in ids]
		self._apply_marker_change()
		return True

	def _undo(self, event=None):
		if not self._undo_stack:
			wx.MessageBox('Nothing to undo.', 'Undo', wx.OK | wx.ICON_INFORMATION)
			return
		self.markers = self._undo_stack.pop()
		self._apply_marker_change()

	def _on_marker_activated(self, event):
		i = event.GetIndex()
		try:
			f = int(self.marker_list.GetItemText(i, 1))
		except ValueError:
			return
		self._stop_play()
		self._seek(f)

	def _jump_risk(self, direction: int):
		bands = sorted(self._events_for_kind(), key=lambda e: e.start_frame)
		bands = [e for e in bands if e.risk_score >= self.min_risk]
		if not bands:
			return
		if direction > 0:
			for e in bands:
				if e.start_frame > self.frame:
					self._stop_play()
					self._seek(e.end_frame)  # jump near separation
					return
			self._seek(bands[0].start_frame)
		else:
			for e in reversed(bands):
				if e.end_frame < self.frame:
					self._stop_play()
					self._seek(e.end_frame)
					return
			self._seek(bands[-1].start_frame)

	def _on_key(self, event):
		key = event.GetKeyCode()
		mods = event.GetModifiers()
		if key == wx.WXK_SPACE:
			self._toggle_play()
		elif key == wx.WXK_LEFT:
			self._nudge(-1)
		elif key == wx.WXK_RIGHT:
			self._nudge(1)
		elif key in (ord('s'), ord('S')) and not (mods & wx.MOD_CONTROL):
			self._mark_swap()
		elif key in (wx.WXK_DELETE, wx.WXK_BACK):
			self._delete_selected_marker()
		elif key in (ord('z'), ord('Z')) and (mods & wx.MOD_CONTROL):
			self._undo()
		elif key in (ord('u'), ord('U')) and not (mods & wx.MOD_CONTROL):
			self._undo()
		elif key in (ord('r'), ord('R')) and not (mods & wx.MOD_CONTROL):
			self._remove_at_current_frame()
		else:
			event.Skip()

	def _on_done(self, event):
		self._stop_play()
		print(f'Finalizing {len(self.markers)} switch marker(s)...', flush=True)
		finalize_switch_annotations(
			self.review_dir,
			self.markers,
			events=self.events,
			export_samples=True,
		)
		self.EndModal(wx.ID_OK)

	def _on_cancel(self, event):
		self._stop_play()
		# leave switches as they were on disk before session? overwrite only on Done
		self.EndModal(wx.ID_CANCEL)

	def _on_close(self, event):
		self._stop_play()
		if self._cap is not None:
			self._cap.release()
			self._cap = None
		event.Skip()

	def Destroy(self):
		self._stop_play()
		if self._cap is not None:
			self._cap.release()
			self._cap = None
		return super().Destroy()

	def get_decisions(self) -> List[ReviewDecision]:
		return [m.to_review_decision() for m in load_switches(self.review_dir)]


# Back-compat names
IdReviewDialog = FullVideoIdReviewDialog


def run_review_dialog(parent, review_dir: str, events: Optional[List[ContactEvent]] = None) -> List[ReviewDecision]:
	'''
	Open full-video timeline review. On OK, switches are finalized on disk.
	Returns decisions derived from switches (empty if cancelled).
	'''
	print(f'Opening full-video ID review ({len(events or load_events(review_dir))} risk band(s))...', flush=True)
	dlg = FullVideoIdReviewDialog(parent, review_dir, events=events)
	try:
		result = dlg.ShowModal()
		if result == wx.ID_OK:
			decisions = dlg.get_decisions()
		else:
			decisions = []
			print('ID review cancelled — no new remaps applied from this session.', flush=True)
	finally:
		dlg.Destroy()
	print(f'ID review closed. Switch decisions: {len(decisions)}', flush=True)
	return decisions
