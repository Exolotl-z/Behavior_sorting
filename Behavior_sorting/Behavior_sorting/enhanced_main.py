import cv2
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime

# Merge gap: bouts separated by <= this many seconds are merged
MERGE_GAP = 0.2  # 0.2 s

# Height (pixels) of the timeline bar under the video
TIMELINE_HEIGHT = 60
WINDOW_COMBINED = "Behavior Annotator"

# Playback speed control (multiplier of original speed)
PLAYBACK_SPEED = 1.0  # 1.0 = normal speed


def merge_intervals(intervals, max_gap=0.0):
    """
    Merge intervals that overlap OR are separated by <= max_gap seconds.
    intervals: list of [start, end]
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        # Merge if overlapping or within max_gap
        if start <= last_end + max_gap:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])

    return merged


class BehaviorAnnotator:
    """
    Enhanced version with multi-behavior support
    """

    def __init__(
        self,
        video_path,
        animal_id,
        session_id,
        behaviors_list,  # List of dicts {'name':, 'key':}
        user_fps=None,
        output_dir=None
    ):
        self.video_path = video_path
        self.animal_id = animal_id
        self.session_id = session_id
        self.output_dir = output_dir or os.path.dirname(video_path)
        
        # Behaviors setup
        self.behaviors = []
        for b in behaviors_list:
            char = b['key'][0].lower()
            self.behaviors.append({
                'name': b['name'],
                'key_char': char,
                'key_code': ord(char)
            })

        # Per-mouse CSV file name
        self.update_output_path()

        # Playback speed control
        self.playback_speed = PLAYBACK_SPEED
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Use video FPS if available, else 30
        self.fps = self.original_fps if self.original_fps > 0 else 30.0

        # Override FPS if user provided one
        if user_fps is not None and user_fps > 0:
            self.fps = float(user_fps)

        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n_frames / self.fps if self.fps > 0 else None

        # Video size
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # Scoring window (absolute times)
        self.scoring_start = 0.0
        self.scoring_end = self.duration if self.duration is not None else 0.0
        self.scoring_locked = False

        print(f"Loaded video: {video_path}")
        print(f"Original FPS: {self.original_fps:.2f}")
        print(f"Using FPS: {self.fps:.2f}")
        print(f"Video duration: {self.duration:.2f} s")
        print(f"Playback speed: {self.playback_speed}x (default)")
        print(f"Initial window: {self.scoring_start:.2f}–{self.scoring_end:.2f} s")
        for b in self.behaviors:
            print(f"Behavior: {b['name']} -> Key: {b['key_char']}")
        print(f"Output file: {self.output_csv}")

        self.frame_idx = 0
        self.current_time = 0.0  # absolute seconds

        # State tracking
        self.active_starts = {} # behavior_name -> start_time
        self.intervals = []  # list of {'start':, 'end':, 'behavior':}
        
        # Colors for different behaviors
        self.behavior_colors = {}
        color_palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, b in enumerate(self.behaviors):
            self.behavior_colors[b['name']] = color_palette[i % len(color_palette)]

        # Timeline bar image (same width as video)
        self.timeline_img = np.zeros(
            (TIMELINE_HEIGHT, self.video_width, 3), dtype=np.uint8
        )

        # Scrubbing state
        self.scrubbing = False

        # Control flags
        self.paused = False
        self.quit_requested = False

        # GUI handles
        self.gui_root = None
        self.gui_time_label = None
        self.gui_scoring_label = None
        self.duration_min_var = None
        self.status_var = None
        self.progress_var = None
        self.speed_label = None
        self.speed_slider = None

        # Last frame (for display when paused/scrubbing)
        self.last_frame = None

        # Single combined window (video + timeline)
        cv2.namedWindow(WINDOW_COMBINED, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_COMBINED, self.on_mouse)

    def update_output_path(self):
        filename = f"{self.animal_id}_{self.session_id}_behaviors.csv"
        self.output_csv = os.path.join(self.output_dir, filename)

    def current_time_rel(self):
        """
        Return time relative to scoring_start (0 at start) once locked.
        Before locking, this just returns absolute time.
        """
        if not self.scoring_locked:
            return self.current_time
        # Clamp to scoring window
        t = max(self.scoring_start, min(self.current_time, self.scoring_end))
        return t - self.scoring_start

    def _effective_duration(self):
        if self.scoring_end is None:
            return 1.0
        return max(1e-6, self.scoring_end - self.scoring_start)

    def time_to_x(self, t):
        """
        Map absolute time t (s) to x coordinate on the timeline,
        using the scoring window [scoring_start, scoring_end].
        """
        if self.scoring_end is None:
            return 0
        t = max(self.scoring_start, min(self.scoring_end, t))
        frac = (t - self.scoring_start) / self._effective_duration()
        return int(frac * (self.video_width - 1))

    def x_to_time(self, x):
        """
        Map x coordinate on the timeline to absolute time (s),
        within [scoring_start, scoring_end].
        """
        if self.scoring_end is None:
            return 0.0
        x = max(0, min(self.video_width - 1, x))
        frac = x / (self.video_width - 1)
        t = self.scoring_start + frac * self._effective_duration()
        return t

    def on_mouse(self, event, x, y, flags, param):
        """
        y < video_height   -> click on video (ignored)
        y >= video_height  -> click on timeline (scrubbing / delete)
        """
        if self.duration is None:
            return

        if y < self.video_height:
            return

        tl_x = x

        if event == cv2.EVENT_LBUTTONDOWN:
            self.scrubbing = True
            self._scrub_to_x(tl_x)

        elif event == cv2.EVENT_MOUSEMOVE and self.scrubbing:
            self._scrub_to_x(tl_x)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.scrubbing:
                self.scrubbing = False
                self._scrub_to_x(tl_x)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # delete interval under cursor
            t = self.x_to_time(tl_x)
            to_delete = None
            for i, interval in enumerate(self.intervals):
                if interval['start'] <= t <= interval['end']:
                    to_delete = i
                    break
            if to_delete is not None:
                removed = self.intervals.pop(to_delete)
                print(f"Deleted interval: {removed['behavior']} {removed['start']:.2f}–{removed['end']:.2f} s")
                self.update_status("Interval deleted")

    def _scrub_to_x(self, x):
        """Scrub video to the time corresponding to x on the timeline."""
        t = self.x_to_time(x)
        self.jump_to_time(t)

    def update_timeline_image(self):
        img = np.zeros((TIMELINE_HEIGHT, self.video_width, 3), dtype=np.uint8)

        # Existing intervals
        for interval in self.intervals:
            x0 = self.time_to_x(interval['start'])
            x1 = self.time_to_x(interval['end'])
            color = self.behavior_colors.get(interval['behavior'], (0, 255, 0))
            cv2.rectangle(img, (x0, 10), (x1, TIMELINE_HEIGHT - 30), color, -1)

        # Current active bouts
        for behavior_name, start_time in self.active_starts.items():
            if start_time is not None:
                x0 = self.time_to_x(start_time)
                x1 = self.time_to_x(self.current_time)
                color = self.behavior_colors.get(behavior_name, (0, 0, 255))
                cv2.rectangle(img, (x0, TIMELINE_HEIGHT - 25), (x1, TIMELINE_HEIGHT - 10), color, -1)

        # Current time marker (white line)
        x_cur = self.time_to_x(self.current_time)
        cv2.line(img, (x_cur, 0), (x_cur, TIMELINE_HEIGHT - 1), (255, 255, 255), 2)

        # Border & text
        cv2.rectangle(
            img, (0, 0), (self.video_width - 1, TIMELINE_HEIGHT - 1),
            (200, 200, 200), 1
        )

        # Show relative time when locked, absolute otherwise
        t_display = self.current_time_rel()
        label = "Rel" if self.scoring_locked else "Abs"
        txt = f"{label}: {t_display:6.2f} s"
        cv2.putText(
            img, txt, (10, TIMELINE_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        self.timeline_img = img

        if self.gui_time_label is not None:
            self.gui_time_label.config(
                text=f"Time ({'Relative' if self.scoring_locked else 'Absolute'}): {t_display:6.2f} s"
            )

        # Update progress bar
        if self.progress_var is not None and self.scoring_locked:
            progress = (self.current_time - self.scoring_start) / self._effective_duration() * 100
            self.progress_var.set(progress)

        self.update_scoring_label()

    def jump_to_time(self, t):
        """Jump video to time t (in seconds, clamped to video)."""
        if self.duration is None:
            return
        t = max(0.0, min(self.duration, t))
        frame_index = int(t * self.fps)
        frame_index = max(0, min(self.n_frames - 1, frame_index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(
            frame, (self.video_width, self.video_height),
            interpolation=cv2.INTER_AREA
        )
        self.last_frame = frame
        self.frame_idx = frame_index
        self.current_time = self.frame_idx / self.fps

    def set_scoring_start(self):
        self.scoring_start = max(0.0, min(self.current_time, self.duration))
        self.scoring_locked = False
        print(f"Set scoring start to {self.scoring_start:.2f} s")
        self.update_status(f"Start time: {self.scoring_start:.2f} s")

    def lock_scoring_window_and_start(self):
        """
        Use start + N minutes as end, lock window, and jump to start.
        """
        if self.duration_min_var is None:
            duration_min = 10.0
        else:
            try:
                duration_min = float(self.duration_min_var.get().strip() or "5")
            except ValueError:
                duration_min = 10.0
        if duration_min <= 0:
            duration_min = 10.0

        proposed_end = self.scoring_start + duration_min * 60.0
        self.scoring_end = min(self.duration, proposed_end)

        if self.scoring_end <= self.scoring_start + 1e-3:
            messagebox.showerror(
                "Invalid Window",
                "Scoring window is too short.\nTry setting start earlier or increasing duration."
            )
            return

        self.scoring_locked = True
        print(f"Locked scoring window: {self.scoring_start:.2f}–{self.scoring_end:.2f} s "
              f"(duration {self.scoring_end - self.scoring_start:.2f} s)")
        self.update_status("Window locked, start annotation")
        self.jump_to_time(self.scoring_start)
        self.paused = False

    def update_scoring_label(self):
        if self.gui_scoring_label is None:
            return
        status = "LOCKED" if self.scoring_locked else "Not Locked"
        length = self.scoring_end - self.scoring_start
        txt = (f"Analysis Window: {self.scoring_start:.2f}–{self.scoring_end:.2f} s "
               f"(dur {length:.2f} s) [{status}]")
        self.gui_scoring_label.config(text=txt)

    def update_status(self, message):
        """Update status bar message"""
        if self.status_var is not None:
            self.status_var.set(message)

    def toggle_behavior(self, behavior_name):
        if not self.scoring_locked:
            messagebox.showinfo(
                "Window Not Locked",
                "Please complete these steps:\n"
                "1) Scrub/play to start time\n"
                "2) Click 'Set start time'\n"
                "3) Enter duration (min)\n"
                "4) Click 'Start annotation (lock)'"
            )
            return

        if behavior_name not in self.active_starts or self.active_starts[behavior_name] is None:
            self.active_starts[behavior_name] = self.current_time
            print(f"Start {behavior_name} at {self.current_time:.2f} s")
            self.update_status(f"{behavior_name} started")
        else:
            start_time = self.active_starts[behavior_name]
            end_time = self.current_time
            if end_time > start_time:
                self.intervals.append({
                    'start': start_time,
                    'end': end_time,
                    'behavior': behavior_name
                })
                duration = end_time - start_time
                print(f"End {behavior_name} at {end_time:.2f} s, duration {duration:.2f} s")
                self.update_status(f"{behavior_name} ended ({duration:.2f} s)")
            self.active_starts[behavior_name] = None

    def toggle_pause(self):
        self.paused = not self.paused
        status = "Paused" if self.paused else "Playing"
        print(status)
        self.update_status(status)

    def jump_back_5(self):
        self.jump_to_time(self.current_time - 5.0)
        print(f"Jumped back to {self.current_time:.2f} s")
        self.update_status(f"Jumped to {self.current_time:.2f} s")

    def jump_forward_5(self):
        self.jump_to_time(self.current_time + 5.0)
        print(f"Jumped forward to {self.current_time:.2f} s")
        self.update_status(f"Jumped to {self.current_time:.2f} s")

    def update_playback_speed(self, value):
        """Update playback speed from slider"""
        self.playback_speed = float(value)
        if self.speed_label:
            self.speed_label.config(text=f"{self.playback_speed:.2f}x")
        self.update_status(f"Speed: {self.playback_speed:.2f}x")
    
    def set_speed_preset(self, speed):
        """Set playback speed to preset value"""
        self.playback_speed = speed
        if hasattr(self, 'speed_slider'):
            self.speed_slider.set(speed)
        if self.speed_label:
            self.speed_label.config(text=f"{speed:.2f}x")
        self.update_status(f"Speed: {speed:.2f}x")
    
    def request_quit(self):
        self.quit_requested = True
    
    def load_next_video(self):
        """Load next video for continuous annotation with metadata update"""
        filename = filedialog.askopenfilename(
            title="Select Next Video",
            filetypes=[
                ("Video files", "*.avi *.mp4 *.mov *.mkv"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        if not filename:
            return
        
        # Save current work if there are intervals
        if self.intervals:
            answer = messagebox.askyesno(
                "Save Current Work?",
                "Do you want to save current annotations before loading new video?"
            )
            if answer:
                self.save_csv()
        
        # Metadata update dialog
        meta_dialog = tk.Toplevel(self.gui_root)
        meta_dialog.title("Update Metadata")
        meta_dialog.geometry("300x200")
        meta_dialog.transient(self.gui_root)
        meta_dialog.grab_set()

        tk.Label(meta_dialog, text="Animal ID:").pack(pady=5)
        aid_entry = tk.Entry(meta_dialog)
        aid_entry.insert(0, self.animal_id)
        aid_entry.pack()

        tk.Label(meta_dialog, text="Session ID:").pack(pady=5)
        sid_entry = tk.Entry(meta_dialog)
        sid_entry.insert(0, self.session_id)
        sid_entry.pack()

        def confirm_meta():
            self.animal_id = aid_entry.get().strip()
            self.session_id = sid_entry.get().strip()
            
            # If output folder was the same as previous video folder, update it to new video folder
            if self.video_path and os.path.dirname(self.video_path) == self.output_dir:
                self.output_dir = os.path.dirname(filename)
            
            self.update_output_path()
            print(f"Metadata updated: Animal={self.animal_id}, Session={self.session_id}")
            print(f"New Output CSV: {self.output_csv}")
            
            meta_dialog.destroy()
            self._do_load_video(filename)

        tk.Button(meta_dialog, text="Load Video", command=confirm_meta).pack(pady=20)
        self.gui_root.wait_window(meta_dialog)

    def _do_load_video(self, filename):
        # Reset for new video
        try:
            # Release old video
            self.cap.release()
            
            # Open new video
            self.cap = cv2.VideoCapture(filename)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video: {filename}")
                return
            
            self.video_path = filename
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = self.original_fps if self.original_fps > 0 else 30.0
            self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.n_frames / self.fps if self.fps > 0 else None
            
            # Reset state
            self.frame_idx = 0
            self.current_time = 0.0
            self.scoring_start = 0.0
            self.scoring_end = self.duration if self.duration is not None else 0.0
            self.scoring_locked = False
            self.active_starts = {}
            self.intervals = []
            self.paused = False
            
            # Update display
            self.jump_to_time(0.0)
            self.update_status(f"Loaded: {os.path.basename(filename)}")
            
            # Update project info labels if they exist
            if hasattr(self, 'aid_label'): self.aid_label.config(text=f"Animal ID: {self.animal_id}")
            if hasattr(self, 'sid_label'): self.sid_label.config(text=f"Session ID: {self.session_id}")
            if hasattr(self, 'out_label'): self.out_label.config(text=f"Output: {os.path.basename(self.output_csv)}")

            print(f"Loaded new video: {filename}")
            print(f"Duration: {self.duration:.2f} s")
            print(f"Output: {self.output_csv}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video:\n{str(e)}")

    def save_csv(self):
        if not self.intervals:
            messagebox.showinfo("No Data", "No intervals recorded.")
            return

        # Sort and merge intervals per behavior
        merged_intervals = []
        behaviors_present = set(i['behavior'] for i in self.intervals)
        for behav in behaviors_present:
            b_ints = [[i['start'], i['end']] for i in self.intervals if i['behavior'] == behav]
            b_merged = merge_intervals(b_ints, MERGE_GAP)
            for m in b_merged:
                merged_intervals.append({'start': m[0], 'end': m[1], 'behavior': behav})

        data_rows = []
        for interval in merged_intervals:
            s2 = max(self.scoring_start, interval['start'])
            e2 = min(self.scoring_end, interval['end'])
            if e2 > s2:
                start_rel = s2 - self.scoring_start
                end_rel = e2 - self.scoring_start
                data_rows.append({
                    "animal_id": self.animal_id,
                    "session_id": self.session_id,
                    "behavior": interval['behavior'],
                    "start_s": round(start_rel, 2),
                    "end_s": round(end_rel, 2),
                    "duration_s": round(end_rel - start_rel, 2)
                })

        if not data_rows:
            messagebox.showinfo(
                "No Data in Window",
                "No intervals fall inside the scoring window."
            )
            return

        df = pd.DataFrame(data_rows)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        # Decide overwrite when file exists
        if os.path.exists(self.output_csv):
            answer = messagebox.askyesno(
                "File Exists",
                f"File '{os.path.basename(self.output_csv)}' already exists.\n\nOverwrite?"
            )
            if not answer:
                return
        
        df.to_csv(self.output_csv, index=False)
        msg = f"Saved {len(df)} intervals to:\n{self.output_csv}"
        print(msg)
        messagebox.showinfo("Saved", msg)
        self.update_status(f"Saved {len(df)} intervals")

    def show_visualization(self):
        """显示数据可视化窗口 (Multi-behavior support)"""
        if not self.intervals:
            messagebox.showinfo("无数据", "请先标记一些行为区间后再查看分析。")
            return

        # Prepare data (Sort and merge per behavior)
        merged_intervals = []
        behaviors_present = set(i['behavior'] for i in self.intervals)
        for behav in behaviors_present:
            b_ints = [[i['start'], i['end']] for i in self.intervals if i['behavior'] == behav]
            b_merged = merge_intervals(b_ints, MERGE_GAP)
            for m in b_merged:
                merged_intervals.append({'start': m[0], 'end': m[1], 'behavior': behav})

        data_rows = []
        for interval in merged_intervals:
            s2 = max(self.scoring_start, interval['start'])
            e2 = min(self.scoring_end, interval['end'])
            if e2 > s2:
                data_rows.append({
                    "start_s": s2 - self.scoring_start,
                    "end_s": e2 - self.scoring_start,
                    "behavior": interval['behavior']
                })

        if not data_rows:
            messagebox.showinfo("窗口内无数据", "分析窗口内没有标记区间。")
            return

        df = pd.DataFrame(data_rows)
        df["duration_s"] = df["end_s"] - df["start_s"]

        # Create visualization window
        viz_window = tk.Toplevel(self.gui_root)
        viz_window.title(f"数据分析 - {self.animal_id}")
        viz_window.geometry("900x750")

        # Create notebook for tabs
        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="统计摘要")

        stats_text = tk.Text(stats_frame, wrap='word', font=('Arial', 10), padx=10, pady=10)
        stats_text.pack(fill='both', expand=True)

        window_duration = self.scoring_end - self.scoring_start
        
        summary_lines = [
            "═══════════════════════════════════════════════",
            "          行为标记统计分析报告",
            "═══════════════════════════════════════════════",
            f"动物ID: {self.animal_id}",
            f"会话ID: {self.session_id}",
            f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析窗口时长: {window_duration:.2f} s",
            "─────────────────────────────────────────────"
        ]

        # Stats per behavior
        for behav in df['behavior'].unique():
            b_df = df[df['behavior'] == behav]
            total_dur = b_df['duration_s'].sum()
            count = len(b_df)
            percentage = (total_dur / window_duration) * 100 if window_duration > 0 else 0
            
            summary_lines.extend([
                f"行为: {behav}",
                f"  标记次数: {count}",
                f"  总持续时间: {total_dur:.2f} s",
                f"  平均持续时间: {b_df['duration_s'].mean():.2f} s",
                f"  行为占比: {percentage:.2f}%",
                "─────────────────────────────────────────────"
            ])

        stats_text.insert('1.0', "\n".join(summary_lines))
        stats_text.config(state='disabled')

        # Tab 2: Timeline
        timeline_frame = ttk.Frame(notebook)
        notebook.add(timeline_frame, text="时间线")

        fig_tl = Figure(figsize=(8, 6), dpi=100)
        ax_tl = fig_tl.add_subplot(111)
        
        behaviors = df['behavior'].unique()
        for i, behav in enumerate(behaviors):
            b_df = df[df['behavior'] == behav]
            color = self.behavior_colors.get(behav, (0, 1, 0))
            # Convert OpenCV color (BGR) to normalized RGB for matplotlib
            if isinstance(color, tuple):
                plt_color = (color[2]/255, color[1]/255, color[0]/255)
            else:
                plt_color = color

            for _, row in b_df.iterrows():
                ax_tl.barh(i, row['duration_s'], left=row['start_s'], height=0.6, 
                        color=plt_color, alpha=0.7, edgecolor='black')
        
        ax_tl.set_xlabel('时间 (秒)', fontsize=12)
        ax_tl.set_yticks(range(len(behaviors)))
        ax_tl.set_yticklabels(behaviors)
        ax_tl.set_title('行为发生时间线', fontsize=14, fontweight='bold')
        ax_tl.set_xlim(0, window_duration)
        ax_tl.grid(True, alpha=0.3, axis='x')
        fig_tl.tight_layout()

        canvas_tl = FigureCanvasTkAgg(fig_tl, timeline_frame)
        canvas_tl.draw()
        canvas_tl.get_tk_widget().pack(fill='both', expand=True)

        # Tab 3: Distribution (BoxPlot)
        box_frame = ttk.Frame(notebook)
        notebook.add(box_frame, text="分布对比")

        fig_box = Figure(figsize=(8, 6), dpi=100)
        ax_box = fig_box.add_subplot(111)
        
        data_to_plot = [df[df['behavior'] == b]['duration_s'].values for b in behaviors]
        bp = ax_box.boxplot(data_to_plot, vert=True, patch_artist=True, labels=behaviors)
        
        for i, patch in enumerate(bp['boxes']):
            color = self.behavior_colors.get(behaviors[i], (0, 255, 0))
            plt_color = (color[2]/255, color[1]/255, color[0]/255)
            patch.set_facecolor(plt_color)
            patch.set_alpha(0.5)

        ax_box.set_ylabel('持续时间 (秒)', fontsize=12)
        ax_box.set_title('各行为持续时间分布', fontsize=14, fontweight='bold')
        ax_box.grid(True, alpha=0.3, axis='y')
        fig_box.tight_layout()

        canvas_box = FigureCanvasTkAgg(fig_box, box_frame)
        canvas_box.draw()
        canvas_box.get_tk_widget().pack(fill='both', expand=True)

    def build_gui(self):
        root = tk.Tk()
        root.title("Behavior Annotator - Control Panel")
        root.geometry("450x680")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Info section
        info_frame = ttk.LabelFrame(main_frame, text="Project Info", padding="10")
        info_frame.pack(fill='x', pady=(0, 10))

        self.aid_label = ttk.Label(info_frame, text=f"Animal ID: {self.animal_id}", font=('Arial', 10))
        self.aid_label.pack(anchor='w')
        self.sid_label = ttk.Label(info_frame, text=f"Session ID: {self.session_id}", font=('Arial', 10))
        self.sid_label.pack(anchor='w')
        self.out_label = ttk.Label(info_frame, text=f"Output: {os.path.basename(self.output_csv)}", font=('Arial', 9))
        self.out_label.pack(anchor='w')

        # Time display
        time_frame = ttk.Frame(main_frame)
        time_frame.pack(fill='x', pady=(0, 5))
        
        time_label = ttk.Label(time_frame, text="Time: 0.00 s", font=('Arial', 11, 'bold'))
        time_label.pack()
        self.gui_time_label = time_label

        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill='x', pady=(0, 10))

        # Scoring window info
        scoring_label = ttk.Label(main_frame, text="", font=('Arial', 9))
        scoring_label.pack(pady=(0, 10))
        self.gui_scoring_label = scoring_label
        self.update_scoring_label()

        # Scoring window controls
        window_frame = ttk.LabelFrame(main_frame, text="Analysis Window", padding="10")
        window_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(window_frame, text="Set Start = Current Time",
                  command=self.set_scoring_start).pack(fill='x', pady=2)

        duration_frame = ttk.Frame(window_frame)
        duration_frame.pack(fill='x', pady=5)
        ttk.Label(duration_frame, text="Duration (min):").pack(side='left')
        self.duration_min_var = tk.StringVar(value="10")
        ttk.Entry(duration_frame, textvariable=self.duration_min_var, width=8).pack(side='left', padx=5)

        ttk.Button(window_frame, text="Start Annotation (Lock)",
                  command=self.lock_scoring_window_and_start).pack(fill='x', pady=2)

        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill='x', pady=(0, 10))

        # Behavior buttons
        behav_frame = ttk.Frame(control_frame)
        behav_frame.pack(fill='x', pady=5)
        for b in self.behaviors:
            btn = ttk.Button(behav_frame, text=f"{b['name']} ({b['key_char']})",
                            command=lambda name=b['name']: self.toggle_behavior(name))
            btn.pack(fill='x', pady=1)

        ttk.Button(control_frame, text="Pause/Resume (Space)",
                  command=self.toggle_pause).pack(fill='x', pady=2)

        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill='x', pady=2)
        ttk.Button(nav_frame, text="Back 5s (J)", command=self.jump_back_5).pack(side='left', fill='x', expand=True, padx=(0, 2))
        ttk.Button(nav_frame, text="Forward 5s (L)", command=self.jump_forward_5).pack(side='left', fill='x', expand=True, padx=(2, 0))
        
        # Playback speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill='x', pady=5)
        ttk.Label(speed_frame, text="Playback Speed:").pack(side='left')
        self.speed_label = ttk.Label(speed_frame, text=f"{self.playback_speed:.2f}x", font=('Arial', 9, 'bold'))
        self.speed_label.pack(side='left', padx=5)
        
        speed_slider = ttk.Scale(speed_frame, from_=0.1, to=2.0, orient='horizontal',
                                command=self.update_playback_speed)
        speed_slider.set(self.playback_speed)
        speed_slider.pack(side='left', fill='x', expand=True, padx=5)
        
        # Speed preset buttons
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(fill='x', pady=2)
        ttk.Button(preset_frame, text="0.25x", width=6,
                  command=lambda: self.set_speed_preset(0.25)).pack(side='left', padx=1)
        ttk.Button(preset_frame, text="0.5x", width=6,
                  command=lambda: self.set_speed_preset(0.5)).pack(side='left', padx=1)
        ttk.Button(preset_frame, text="1.0x", width=6,
                  command=lambda: self.set_speed_preset(1.0)).pack(side='left', padx=1)
        ttk.Button(preset_frame, text="1.5x", width=6,
                  command=lambda: self.set_speed_preset(1.5)).pack(side='left', padx=1)
        ttk.Button(preset_frame, text="2.0x", width=6,
                  command=lambda: self.set_speed_preset(2.0)).pack(side='left', padx=1)
        
        self.speed_slider = speed_slider

        # File and Save operations
        action_frame = ttk.LabelFrame(main_frame, text="File Operations", padding="10")
        action_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(action_frame, text="Save CSV (C)",
                  command=self.save_csv).pack(fill='x', pady=2)
        
        ttk.Button(action_frame, text="Load Next Video (N)",
                  command=self.load_next_video).pack(fill='x', pady=2)

        ttk.Button(action_frame, text="Quit (Q)",
                  command=self.request_quit).pack(fill='x', pady=2)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', side='bottom')
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief='sunken', anchor='w')
        status_label.pack(fill='x')

        # Keyboard shortcuts
        for b in self.behaviors:
            char = b['key_char']
            root.bind(f"<Key-{char}>", lambda e, name=b['name']: self.toggle_behavior(name))
            root.bind(f"<Key-{char.upper()}>", lambda e, name=b['name']: self.toggle_behavior(name))
            
        root.bind("<space>", lambda e: self.toggle_pause())
        root.bind("<Key-j>", lambda e: self.jump_back_5())
        root.bind("<Key-J>", lambda e: self.jump_back_5())
        root.bind("<Key-l>", lambda e: self.jump_forward_5())
        root.bind("<Key-L>", lambda e: self.jump_forward_5())
        root.bind("<Key-c>", lambda e: self.save_csv())
        root.bind("<Key-C>", lambda e: self.save_csv())
        root.bind("<Key-n>", lambda e: self.load_next_video())
        root.bind("<Key-N>", lambda e: self.load_next_video())
        root.bind("<Key-q>", lambda e: self.request_quit())
        root.bind("<Key-Q>", lambda e: self.request_quit())

        self.gui_root = root

    def run(self):
        self.build_gui()
        self.jump_to_time(0.0)
        self.update_status("Ready - Set analysis window")

        # Apply playback speed multiplier dynamically
        delay_ms = int((1000 / self.fps) / self.playback_speed) if self.fps > 0 else 30

        while True:
            # keep Tk GUI alive
            if self.gui_root is not None:
                try:
                    self.gui_root.update_idletasks()
                    self.gui_root.update()
                except tk.TclError:
                    self.gui_root = None

            if self.quit_requested:
                print("Quit requested")
                break

            if not self.paused and not self.scrubbing:
                ret, frame = self.cap.read()
                if not ret:
                    # Video ended - pause instead of quit
                    print("Video ended - paused")
                    self.paused = True
                    self.update_status("Video ended - Ready for next")
                    # Don't break, continue loop
                    continue
                self.frame_idx += 1
                self.current_time = self.frame_idx / self.fps

                frame = cv2.resize(
                    frame, (self.video_width, self.video_height),
                    interpolation=cv2.INTER_AREA
                )
                
                t_disp = self.current_time_rel()
                label = "Rel" if self.scoring_locked else "Abs"
                overlay_text1 = f"{label}: {t_disp:6.2f} s"
                
                # Check active behaviors
                active_behaviors = [name for name, start in self.active_starts.items() if start is not None]
                overlay_text2 = f"ACTIVE: {', '.join(active_behaviors)}" if active_behaviors else "Ready"
                color = (0, 0, 255) if active_behaviors else (0, 255, 0)

                cv2.putText(frame, overlay_text1, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, overlay_text2, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(
                    frame,
                    "Drag timeline to scrub | Right-click to delete",
                    (10, self.video_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )

                self.last_frame = frame

                if self.scoring_locked and self.current_time > self.scoring_end:
                    print("Reached end of scoring window")
                    self.paused = True
                    self.update_status("Reached window end")

            self.update_timeline_image()

            if self.last_frame is not None:
                combined = np.vstack([self.last_frame, self.timeline_img])
            else:
                blank_video = np.zeros(
                    (self.video_height, self.video_width, 3), dtype=np.uint8
                )
                combined = np.vstack([blank_video, self.timeline_img])

            cv2.imshow(WINDOW_COMBINED, combined)

            # Recalculate delay based on current playback speed
            delay_ms = int((1000 / self.fps) / self.playback_speed) if self.fps > 0 else 30
            
            key = cv2.waitKey(delay_ms if not self.paused else 50) & 0xFF
            
            # Check behavior keys
            for b in self.behaviors:
                if key == b['key_code']:
                    self.toggle_behavior(b['name'])
            
            if key == ord(" "):
                self.toggle_pause()
            elif key == ord("j"):
                self.jump_back_5()
            elif key == ord("l"):
                self.jump_forward_5()
            elif key == ord("c"):
                self.save_csv()
            elif key == ord("q"):
                self.request_quit()

        # close any active intervals if needed
        for name, start_time in self.active_starts.items():
            if start_time is not None:
                end_time = self.current_time
                if end_time > start_time:
                    self.intervals.append({
                        'start': start_time,
                        'end': end_time,
                        'behavior': name
                    })
                    print(f"Video ended, closing {name}: {start_time:.2f}–{end_time:.2f} s")

        self.cap.release()
        cv2.destroyAllWindows()
        if self.gui_root is not None:
            self.gui_root.destroy()
        print("Program terminated")


class ConfigWindow:
    """Enhanced configuration window with multi-behavior support"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Behavior Annotator - Setup")
        self.root.geometry("650x700")
        
        style = ttk.Style()
        style.theme_use('clam')

        self.video_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.animal_id_var = tk.StringVar(value="M01")
        self.session_id_var = tk.StringVar(value="baseline")
        self.frame_rate_var = tk.StringVar(value="0")
        self.num_behaviors_var = tk.IntVar(value=1)

        # Main container
        self.main_scroll = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.main_scroll.yview)
        self.scrollable_frame = ttk.Frame(self.main_scroll, padding="20")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_scroll.configure(scrollregion=self.main_scroll.bbox("all"))
        )

        self.main_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_scroll.configure(yscrollcommand=self.scrollbar.set)

        self.main_scroll.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(self.scrollable_frame, text="Behavior Annotator", 
                  font=('Arial', 16, 'bold')).pack(pady=(0, 20))

        # Input frame
        input_frame = ttk.Frame(self.scrollable_frame)
        input_frame.pack(fill='both', expand=True)

        # Video file
        ttk.Label(input_frame, text="Video File:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.video_path_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)

        # Output dir
        ttk.Label(input_frame, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # Metadata
        ttk.Label(input_frame, text="Animal ID:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.animal_id_var, width=20).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(input_frame, text="Session ID:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.session_id_var, width=20).grid(row=3, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(input_frame, text="Frame Rate (0=auto):").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.frame_rate_var, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Behaviors section
        behav_header = ttk.LabelFrame(self.scrollable_frame, text="Behaviors Configuration", padding="10")
        behav_header.pack(fill='x', pady=10)

        count_frame = ttk.Frame(behav_header)
        count_frame.pack(fill='x')
        ttk.Label(count_frame, text="Number of Behaviors:").pack(side='left')
        self.spin = tk.Spinbox(count_frame, from_=1, to=10, textvariable=self.num_behaviors_var, 
                               width=5, command=self.update_behavior_fields)
        self.spin.pack(side='left', padx=10)

        self.behav_list_frame = ttk.Frame(behav_header)
        self.behav_list_frame.pack(fill='x', pady=10)
        
        self.behavior_entries = [] # List of (name_var, key_var)
        self.update_behavior_fields()

        # Buttons
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(pady=20, fill='x')
        
        ttk.Button(btn_frame, text="Launch Annotator", 
                  command=self.start_annotation).pack(side='left', fill='x', expand=True, padx=5)
        
        ttk.Button(btn_frame, text="Batch Analyzer", 
                  command=self.launch_batch_analyzer).pack(side='left', fill='x', expand=True, padx=5)

    def update_behavior_fields(self):
        for widget in self.behav_list_frame.winfo_children():
            widget.destroy()
        
        self.behavior_entries = []
        n = self.num_behaviors_var.get()
        
        # Headers
        ttk.Label(self.behav_list_frame, text="Behavior Name", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5)
        ttk.Label(self.behav_list_frame, text="Toggle Key", font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5)

        default_names = ["Grooming", "Sniffing", "Rearing", "Walking", "Still"]
        default_keys = ["g", "s", "r", "w", "f"]

        for i in range(n):
            name_var = tk.StringVar(value=default_names[i] if i < len(default_names) else f"Behav_{i+1}")
            key_var = tk.StringVar(value=default_keys[i] if i < len(default_keys) else str(i))
            
            ttk.Entry(self.behav_list_frame, textvariable=name_var, width=15).grid(row=i+1, column=0, padx=5, pady=2)
            ttk.Entry(self.behav_list_frame, textvariable=key_var, width=5).grid(row=i+1, column=1, padx=5, pady=2)
            
            self.behavior_entries.append((name_var, key_var))

    def browse_video(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(filename))

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def start_annotation(self):
        video_path = self.video_path_var.get().strip()
        animal_id = self.animal_id_var.get().strip()
        session_id = self.session_id_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        try:
            fps = float(self.frame_rate_var.get())
        except ValueError:
            fps = 0

        behaviors_list = []
        for name_var, key_var in self.behavior_entries:
            name = name_var.get().strip()
            key = key_var.get().strip().lower()
            if name and key:
                behaviors_list.append({'name': name, 'key': key})

        if not behaviors_list:
            messagebox.showerror("Error", "Please define at least one behavior.")
            return

        self.root.withdraw()
        try:
            app = BehaviorAnnotator(
                video_path=video_path,
                animal_id=animal_id,
                session_id=session_id,
                behaviors_list=behaviors_list,
                user_fps=fps if fps > 0 else None,
                output_dir=output_dir
            )
            app.run()
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
        finally:
            self.root.deiconify()

    def launch_batch_analyzer(self):
        """Launch batch analysis tool from config window"""
        try:
            import sys
            import importlib.util
            batch_analyzer_path = os.path.join(os.path.dirname(__file__), "batch_analyzer.py")
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
                batch_analyzer_path = os.path.join(base_path, "batch_analyzer.py")
            if not os.path.exists(batch_analyzer_path):
                messagebox.showerror("Error", f"Batch analyzer not found:\n{batch_analyzer_path}")
                return
            spec = importlib.util.spec_from_file_location("batch_analyzer", batch_analyzer_path)
            batch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(batch_module)
            analyzer_root = tk.Toplevel(self.root)
            batch_module.BatchAnalyzer(analyzer_root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch batch analyzer:\n{str(e)}")

def main():
    root = tk.Tk()
    ConfigWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
