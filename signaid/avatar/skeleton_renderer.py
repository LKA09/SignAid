from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Polygon
import numpy as np

from signaid.datasets.mock import SKELETON_EDGES


class SkeletonRenderer:
    def __init__(self, fps: int = 20, edges: tuple[tuple[int, int], ...] = SKELETON_EDGES) -> None:
        self.fps = fps
        self.edges = edges

    @staticmethod
    def load_motion(path: Path) -> tuple[np.ndarray, int]:
        with np.load(path) as data:
            key = "motion" if "motion" in data else data.files[0]
            return np.asarray(data[key], dtype=np.float32), int(data.get("fps", 20))

    def render(self, motion_path: Path, output_path: Path) -> Path:
        motion, fps = self.load_motion(motion_path)
        return self.render_array(motion, output_path, fps=fps)

    def _make_animation(self, motion: np.ndarray, fps: int):
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 3 or motion.shape[-1] != 3 or not len(motion):
            raise ValueError("motion must be non-empty with shape (T, J, 3)")
        fig = plt.figure(figsize=(5.4, 5.4), facecolor="#071426")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, facecolor="#071426")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-0.10, 2.12)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # The motion remains joint-driven internally, but the viewer only sees
        # a filled human character. This makes the pose readable without
        # presenting the implementation skeleton as the final avatar.
        skin = "#d79a78"
        skin_light = "#efb796"
        hair = "#172536"
        shirt = "#247c83"
        shirt_light = "#35aab0"
        trousers = "#203b5a"
        outline = "#10243a"

        ax.add_patch(Ellipse((0, -0.015), 0.72, 0.10, color="#020914", alpha=0.65, zorder=0))
        # Soft backdrop rings keep the silhouette separated from the stage.
        ax.add_patch(Circle((0, 1.10), 0.83, fill=False, lw=1.2, color="#2dd4bf", alpha=0.10, zorder=0))
        ax.add_patch(Circle((0, 1.10), 0.63, fill=False, lw=0.8, color="#79dff2", alpha=0.08, zorder=0))

        def line(color: str, width: float, zorder: int):
            return ax.plot([], [], color=color, lw=width, solid_capstyle="round", zorder=zorder)[0]

        # Legs and shoes.
        leg_shadow = [line(outline, 18, 2) for _ in range(4)]
        legs = [line(trousers, 13, 3) for _ in range(4)]
        shoes = [line("#091321", 15, 4) for _ in range(2)]

        torso_shadow = Polygon(np.zeros((4, 2)), closed=True, facecolor=outline, edgecolor="none", zorder=4)
        torso = Polygon(np.zeros((4, 2)), closed=True, facecolor=shirt, edgecolor=shirt_light, lw=1.5, zorder=5)
        ax.add_patch(torso_shadow)
        ax.add_patch(torso)
        waist = line("#172f49", 4, 6)
        collar_left = line("#c8ffff", 2.2, 7)
        collar_right = line("#c8ffff", 2.2, 7)

        # Arms are layered over the torso so signing near the chest stays visible.
        upper_arm_shadow = [line(outline, 19, 7) for _ in range(4)]
        upper_arms = [line(shirt, 14, 8) for _ in range(2)]
        forearms = [line(skin, 12, 9) for _ in range(2)]

        neck = line(skin, 19, 6)
        ears = [Circle((0, 0), 0.038, facecolor=skin, edgecolor=outline, lw=1, zorder=10) for _ in range(2)]
        face = Circle((0, 0), 0.135, facecolor=skin_light, edgecolor=outline, lw=2, zorder=11)
        hair_cap = Polygon(np.zeros((7, 2)), closed=True, facecolor=hair, edgecolor=outline, lw=1, zorder=12)
        for patch in [*ears, face, hair_cap]:
            ax.add_patch(patch)
        eyes = [Circle((0, 0), 0.010, facecolor="#172536", edgecolor="none", zorder=13) for _ in range(2)]
        eye_lights = [Circle((0, 0), 0.003, facecolor="white", edgecolor="none", zorder=14) for _ in range(2)]
        for patch in [*eyes, *eye_lights]:
            ax.add_patch(patch)
        brows = [line("#4a3029", 2.0, 13) for _ in range(2)]
        nose = line("#b8735f", 1.4, 13)
        mouth = line("#9d4e55", 2.0, 13)

        hand_edges = []
        if motion.shape[1] >= 59:
            hand_edges = [(a, b) for a, b in self.edges if a >= 17 and b >= 17]
        finger_shadow = [line(outline, 7.0, 14) for _ in hand_edges]
        fingers = [line(skin_light, 4.7, 15) for _ in hand_edges]
        palms = [Polygon(np.zeros((5, 2)), closed=True, facecolor=skin_light, edgecolor=outline, lw=1.5, zorder=14) for _ in range(2)]
        for palm in palms:
            ax.add_patch(palm)
        fingertips = [Circle((0, 0), 0.014, facecolor=skin_light, edgecolor=outline, lw=0.7, zorder=16) for _ in range(10)]
        for tip in fingertips:
            ax.add_patch(tip)

        def xy(points: np.ndarray, indices: list[int]) -> np.ndarray:
            return points[indices, :2]

        def set_segment(artist, points: np.ndarray, a: int, b: int):
            artist.set_data([points[a, 0], points[b, 0]], [points[a, 1], points[b, 1]])

        def update(frame: int):
            points = motion[frame]

            # Legs: hips-knees and knees-ankles.
            leg_pairs = [(10, 11), (11, 12), (13, 14), (14, 15)]
            for shadow_artist, artist, (a, b) in zip(leg_shadow, legs, leg_pairs):
                set_segment(shadow_artist, points, a, b)
                set_segment(artist, points, a, b)
            for artist, ankle, direction in zip(shoes, (12, 15), (-1, 1)):
                artist.set_data([points[ankle, 0] - 0.02 * direction, points[ankle, 0] + 0.10 * direction],
                                [points[ankle, 1], points[ankle, 1]])

            shoulder_y = (points[2, 1] + points[5, 1]) / 2
            torso_points = np.array([
                [points[2, 0] - 0.055, shoulder_y + 0.035],
                [points[5, 0] + 0.055, shoulder_y + 0.035],
                [points[13, 0] + 0.055, points[13, 1]],
                [points[10, 0] - 0.055, points[10, 1]],
            ])
            center = torso_points.mean(axis=0)
            torso_shadow.set_xy(center + (torso_points - center) * 1.045)
            torso.set_xy(torso_points)
            waist.set_data([points[10, 0] - 0.04, points[13, 0] + 0.04], [points[10, 1], points[13, 1]])
            collar = points[1, :2]
            collar_left.set_data([collar[0], collar[0] - 0.105], [collar[1] - 0.07, collar[1] - 0.01])
            collar_right.set_data([collar[0], collar[0] + 0.105], [collar[1] - 0.07, collar[1] - 0.01])
            neck.set_data([points[1, 0], points[0, 0]], [points[1, 1], points[0, 1] - 0.09])

            # Two outline segments per arm, then clothing upper arm and skin forearm.
            arm_pairs = [(2, 3), (3, 4), (5, 6), (6, 7)]
            for artist, (a, b) in zip(upper_arm_shadow, arm_pairs):
                set_segment(artist, points, a, b)
            for artist, (a, b) in zip(upper_arms, ((2, 3), (5, 6))):
                set_segment(artist, points, a, b)
            for artist, (a, b) in zip(forearms, ((3, 4), (6, 7))):
                set_segment(artist, points, a, b)

            head = points[0, :2]
            face.center = tuple(head)
            ears[0].center = (head[0] - 0.132, head[1] - 0.005)
            ears[1].center = (head[0] + 0.132, head[1] - 0.005)
            hair_cap.set_xy(np.array([
                [head[0] - .13, head[1] + .015], [head[0] - .105, head[1] + .105],
                [head[0] - .025, head[1] + .145], [head[0] + .07, head[1] + .13],
                [head[0] + .135, head[1] + .065], [head[0] + .125, head[1] + .02],
                [head[0] + .02, head[1] + .07],
            ]))
            for side, eye, light, brow in zip((-1, 1), eyes, eye_lights, brows):
                ex, ey = head[0] + side * .047, head[1] + .018
                eye.center = (ex, ey)
                light.center = (ex - .002, ey + .003)
                brow.set_data([ex - .025, ex + .025], [ey + .035, ey + .038])
            nose.set_data([head[0], head[0] - .009], [head[1] + .005, head[1] - .03])
            mouth.set_data([head[0] - .033, head[0], head[0] + .033],
                           [head[1] - .066, head[1] - .074, head[1] - .064])

            if hand_edges:
                for shadow_artist, artist, (a, b) in zip(finger_shadow, fingers, hand_edges):
                    set_segment(shadow_artist, points, a, b)
                    set_segment(artist, points, a, b)
                palm_indices = ([17, 22, 26, 30, 34], [38, 43, 47, 51, 55])
                for palm, indices in zip(palms, palm_indices):
                    palm.set_xy(xy(points, indices))
                tip_indices = [21, 25, 29, 33, 37, 42, 46, 50, 54, 58]
                for tip, index in zip(fingertips, tip_indices):
                    tip.center = tuple(points[index, :2])

            return [*leg_shadow, *legs, *shoes, torso_shadow, torso, waist,
                    collar_left, collar_right, *upper_arm_shadow, *upper_arms,
                    *forearms, neck, *ears, face, hair_cap, *eyes, *eye_lights,
                    *brows, nose, mouth, *finger_shadow, *fingers, *palms, *fingertips]

        return fig, FuncAnimation(fig, update, frames=len(motion), interval=1000/fps, blit=False)

    def render_array(self, motion: np.ndarray, output_path: Path, fps: int | None = None) -> Path:
        fps = fps or self.fps
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, animation = self._make_animation(motion, fps)
        try:
            if output_path.suffix.lower() == ".mp4" and shutil.which("ffmpeg"):
                animation.save(output_path, writer="ffmpeg", fps=fps, dpi=100)
            else:
                if output_path.suffix.lower() != ".gif":
                    output_path = output_path.with_suffix(".gif")
                animation.save(output_path, writer=PillowWriter(fps=fps), dpi=80)
        finally:
            plt.close(fig)
        return output_path

    def preview(self, motion: np.ndarray) -> None:
        # Interactive preview is intentionally opt-in so server/headless usage stays safe.
        previous = matplotlib.get_backend()
        fig = None
        try:
            plt.switch_backend("TkAgg")
            fig, _animation = self._make_animation(motion, self.fps)
            plt.show()
        finally:
            if fig is not None:
                plt.close(fig)
            plt.switch_backend(previous)
