# -*- coding: utf-8 -*-
"""
===============================================================================
[one フレームワーク版: UR3Dual 左腕 アルミフレーム全自動統合シーケンス]
 (検出 ➔ ピック ➔ ホーム搬送 ➔ 3回接触 ➔ 誤差同定 ➔ 精密設置)
-------------------------------------------------------------------------------
ファイル名: integrated_pick_contact_place.py
対応フレームワーク: one (MuJoCo / pyglet)

【全体の流れ】:
  [V] キー : 撮影待機姿勢 (View Pose) への移動プレビュー (Sim確認後 [Enter] で実機実行)
  [S] キー : カメラ検出位置で静止・姿勢確定 ＆ ピック動作計画
  [1] キー : ピック動作の実行 (アプローチ ➔ 下降 ➔ 把持)
  [2] キー : 把持物体の持ち上げ ➔ 3回接触ホーム姿勢への安全搬送
  [3] キー : 3回接触シーケンス実行 (各角度傾斜 ➔ 鉛直下降 ➔ 反力検知 ➔ 誤差同定 ➔ 設置計画)
  [4] キー : 精密設置動作の実行 (同定結果を反映した配置位置へ搬送・設置・開放)

【確認・補助キー】:
  [Enter] / [Y] : シミュレーション確認完了 ➔ 実機へ送信して動作実行！
  [H] キー      : 待機ホーム姿勢へ移動
  [O] / [C]     : グリッパー手動開閉
  [ESC]         : プレビューキャンセル / 中断 / 非常停止
===============================================================================
"""

import faulthandler
faulthandler.enable()

import os
import sys
import time
import json
import yaml
import socket
import select
import termios
import tty
import threading
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2

# one パッケージパスの解決
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

cam_dir = os.path.join(project_root, '0001_nagahama', 'camera')
if cam_dir not in sys.path:
    sys.path.insert(0, cam_dir)

mesh_dir = os.path.join(project_root, '0001_nagahama', 'meshes')

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
import one.geom.loader as ogl
import one.robots.manipulators.universal_robots.ur3.ur3 as ormuu3
from one.robots.end_effectors.robotiq.rtq2f85.rtq2f85 import Rtq2F85

try:
    import pyglet
    from pyglet.window import key
    from pyglet import shapes, gl
except ImportError:
    pass

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    HAS_UR_RTDE = True
except ImportError:
    HAS_UR_RTDE = False

from detect_aluminum_profile import ProfileDetector

# ===============================================================================
# 定数・パラメータ設定
# ===============================================================================
DEVICE_ID = 12
ROBOT_IP = '10.2.0.50'

TABLE_Z = 1.0940           # 机面天板のワールド Z 座標 [m]
PROFILE_HEIGHT = 0.060
TOOL_OFFSET_FLANGE = np.array([0.0, 0.0, 0.1938])     # アルミフレームの高さ [m] (6cm)
PROFILE_WIDTH = 0.060      # アルミフレームの幅 [m] (6cm)
PROFILE_LENGTH = 0.160     # アルミフレームの長さ [m] (16cm)
TARGET_PLANE_Z = TABLE_Z + PROFILE_HEIGHT  # 上面のワールド Z 座標 (1.1540 m)

# ★ 実機の把持力は常に最大 (255)
GRIPPER_MAX_FORCE = 255

# ===============================================================================
# 実機ロボット移動速度・加速度パラメータ [rad/s, rad/s^2] (低速・安全設定)
# ===============================================================================
SPEED_NORMAL  = 0.18   # 通常移動速度 (待機姿勢・アプローチ・搬送) [rad/s]
ACCEL_NORMAL  = 0.20   # 通常加速度 [rad/s^2]
SPEED_SLOW    = 0.10   # 慎重移動速度 (把持下降・垂直リフト・接地設置) [rad/s]
ACCEL_SLOW    = 0.15   # 慎重加速度 [rad/s^2]
SPEED_CONTACT = 0.02   # 接触同定の低速下降速度 [rad/s]
ACCEL_CONTACT = 0.04   # 接触同定の低速下降加速度 [rad/s^2]

GRIP_DEPTH_M = 0.040       # ピック時の把持深さ (上面から4cm深く掴む)

# 基準把持姿勢 R_BASE (鉛直真下向き, 開閉方向: 世界X軸平行)
R_BASE = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
])

# 把持位置のオフセット:
# 全長 160mm の中心 (80mm) から 35mm オフセット = 底面から 115mm, 上面から 45mm 潜った位置
# これにより CARRY_HOME_Q への搬送時にアルミフレーム下端が鉛直真下を向き、机面上 11.5cm 余裕を確保
GRASP_CENTER_OFFSET_M = 0.035

# UR3Dual 左腕の各種基本姿勢
# ★ 検出用カメラ観察姿勢 (VIEW_HOME_Q): nrs_vr2 と完全一致 (光軸鉛直真下向き)
# 度数法: [-17.7°, -209.6°, +42.6°, +273.9°, +227.6°, -245.8°]
VIEW_HOME_Q = np.deg2rad([-17.7, -209.6, 42.6, 273.9, 227.6, -245.8])
VIEW_POSE_Q = VIEW_HOME_Q

# ★ 3回接触ホームポジション (CONTACT_HOME_Q): nrs_vr2 と完全一致 (机面上 11.5cm 余裕確保)
CONTACT_HOME_Q = np.array([0.4967, -3.2567, 1.8685, 3.4037, 2.2417, -4.0584])
SAFE_HOME_Q = CONTACT_HOME_Q
CARRY_HOME_Q = CONTACT_HOME_Q

PLACE_STANDBY_Q = np.radians([23.86, -195.12, 68.14, 194.97, 229.71, -302.02])

# ===============================================================================
# ロボットモデル生成関数
# ===============================================================================
import one.scene.render_model_primitive as osrmp

def _unwrap_to_prev(target_q, prev_q):
    res = np.copy(target_q)
    for i in range(len(res)):
        diff = res[i] - prev_q[i]
        k = np.round(diff / (2.0 * np.pi))
        res[i] -= k * 2.0 * np.pi
    return res

def build_ur3_dual_lft():
    """one フレームワークで UR3Dual 左腕 + Rtq2F85 モデルを構築 (カップリング円柱完全統合)"""
    base_pos = np.array([0.0, 0.258485281374, 1.61051471863], dtype=np.float32)
    base_rotmat = oum.rotmat_from_euler(-3.0 * np.pi / 4.0, 0.0, 0.0)
    robot = ormuu3.UR3(pos=base_pos, rotmat=base_rotmat)

    flange_lnk = robot.runtime_lnks[-1]

    # ★ フランジとグリッパ間の空洞を完全に埋めるカップリング円柱 (長さ 0.049962m, 半径 37.5mm, リアルなFT300色)
    cyl_rmodel = osrmp.gen_cylinder_rmodel(
        length=0.049962,
        radius=0.0375,
        n_segs=32,
        rgb=np.array([0.22, 0.22, 0.25], dtype=np.float32)
    )
    flange_lnk.add_visual(cyl_rmodel, auto_make_collision=False)

    gripper = Rtq2F85()
    # ★ カップリング角度は 0.0 (実機開閉方向と完全一致: 90度ズレ解消)
    loc_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, 0.049962], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(0.0, 0.0, 0.0),
    )
    robot.mount(gripper, flange_lnk, loc_tf, update=True)
    return robot, gripper


def pick_closest_ik(sols, ref_q):
    """解析解の中から参照関節角に最も近い解を選択"""
    if not sols:
        return None
    best_q = None
    min_dist = float('inf')
    for sol in sols:
        diff = (sol - ref_q + np.pi) % (2.0 * np.pi) - np.pi
        unwrapped = ref_q + diff
        dist = np.linalg.norm(diff)
        if dist < min_dist:
            min_dist = dist
            best_q = unwrapped
    return best_q


# ===============================================================================
# ===============================================================================
# ===============================================================================
# Robotiq 85 グリッパ制御 (URCap ポート 63352 / SID自動認識・確実アクティベート通信)
# ===============================================================================
class RobotiqGripper:
    """
    Robotiq 2F-85 グリッパコントローラ (URCap ポート 63352)
    ・FT300 + Robotiq 構成の Slave ID (SID 9 / 自動検出) に完全対応
    ・SET ACT 0 -> SET ACT 1 -> SET GTO 1 による正規アクティベーション手順
    ・STA 3 (Active) の確実な待機と自己修復 (Auto-recovery)
    ・リアルタイム位置・爪幅取得 (GET POS)
    """
    def __init__(self, robot_ip=ROBOT_IP, port=63352, sid=9, timeout=2.0):
        self.robot_ip = robot_ip
        self.port = port
        self.sid = sid
        self.timeout = timeout
        self._lock = threading.RLock()
        self.is_connected = False
        self._init_gripper()

    def _test_and_detect_sid(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.robot_ip, self.port))
            s.sendall(b"GET SID\n")
            sid_res = s.recv(1024).decode("utf-8").strip()
            s.close()
            
            import re
            m = re.search(r"\d+", sid_res)
            if m:
                detected_sid = int(m.group())
                if detected_sid > 0:
                    self.sid = detected_sid
            return True
        except Exception as e:
            return False

    def _send_cmd(self, cmd_str, retry=2):
        """1コマンドごとに接続・SID指定・送信・切断する短命ソケット方式"""
        with self._lock:
            for attempt in range(retry):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(self.timeout)
                    s.connect((self.robot_ip, self.port))
                    
                    # FT300 / 双腕構成に対応するため必ず対象 SID を指定
                    if self.sid is not None:
                        s.sendall(f"SET SID {self.sid}\n".encode("utf-8"))
                        s.recv(1024)
                    
                    msg = (cmd_str.strip() + "\n").encode("utf-8")
                    s.sendall(msg)
                    res = s.recv(1024).decode("utf-8").strip()
                    s.close()
                    self.is_connected = True
                    return res
                except Exception as e:
                    if attempt == retry - 1:
                        self.is_connected = False
                        if self.robot_ip != "127.0.0.1":
                            print(f"⚠️ グリッパー通信エラー ({cmd_str.strip()}): {e}")
                    time.sleep(0.1)
            return ""

    def ensure_activated(self):
        """グリッパーが動作可能状態 (STA 3) であることを確認し、未初期化なら自動アクティベート"""
        if not self.is_connected:
            return
        sta_res = self._send_cmd("GET STA")
        if "STA 3" not in sta_res and "3" not in sta_res:
            self._activate_sequence()

    def _activate_sequence(self):
        """Robotiq 正規アクティベーションシーケンス (ACT 0 -> ACT 1 + GTO 1 -> STA 3 待機)"""
        print(f"  ・グリッパーのリセット・アクティベーションを開始します... (SID: {self.sid})")
        # 1. まず ACT 0 でリセット (以前のエラーや未初期化状態をクリア)
        self._send_cmd("SET ACT 0")
        self._send_cmd("SET ATR 0")
        time.sleep(0.5)

        # 2. ACT 1 (アクティベート要求) と GTO 1 (動作許可) を送信
        self._send_cmd("SET ACT 1")
        time.sleep(0.2)
        self._send_cmd("SET GTO 1")
        time.sleep(0.2)
        self._send_cmd("SET SPE 255")
        self._send_cmd(f"SET FOR {GRIPPER_MAX_FORCE}")

        # 3. アクティベーション完了 (STA 3) を待機 (最大約 8 秒)
        for i in range(20):
            time.sleep(0.4)
            cur_sta = self._send_cmd("GET STA")
            cur_flt = self._send_cmd("GET FLT")
            if "STA 3" in cur_sta or "3" in cur_sta:
                print(f"  ✓ Robotiq グリッパーのアクティベーション完了！ (状態: {cur_sta}, 準備完了)")
                return True
            if i % 5 == 0:
                print(f"  ... アクティベート待機中 (経過: {i*0.4:.1f}s, 状態: {cur_sta}, 異常: {cur_flt})")

        print("⚠️ グリッパーのアクティベートがタイムアウトしました。URのティーチペンダント側でグリッパーが有効化されているか確認してください。")
        return False

    def _init_gripper(self):
        if not self._test_and_detect_sid():
            print(f"⚠️ Robotiq Gripper ({self.robot_ip}:{self.port}) 接続失敗。シミュレーションモードで継続します。")
            self.is_connected = False
            return

        self.is_connected = True
        print(f"✓ Robotiq Gripper ({self.robot_ip}:{self.port}, SID: {self.sid}) に接続成功！")

        # 起動時のアクティベーション実行
        try:
            self.ensure_activated()
        except Exception as e:
            print(f"⚠️ グリッパー初期化警告: {e}")

    def open(self):
        """グリッパ全開 (85 mm, POS 0)"""
        self.ensure_activated()
        self._send_cmd("SET GTO 1")
        self._send_cmd("SET SPE 255")
        self._send_cmd("SET FOR 50")
        res = self._send_cmd("SET POS 0")
        time.sleep(0.8)
        return res

    def close(self, force=GRIPPER_MAX_FORCE):
        """グリッパ全閉 (0 mm, POS 255)"""
        self.ensure_activated()
        self._send_cmd("SET GTO 1")
        self._send_cmd("SET SPE 255")
        self._send_cmd(f"SET FOR {force}")
        res = self._send_cmd("SET POS 255")
        time.sleep(1.0)
        return res

    def move_to_width(self, width_m, force=GRIPPER_MAX_FORCE, speed=255):
        """指定した把持幅 [m] (0.000 ~ 0.085) にグリッパを移動"""
        self.ensure_activated()
        width_clamped = float(np.clip(width_m, 0.0, 0.085))
        pos_int = int(round((0.085 - width_clamped) / 0.085 * 255.0))
        self._send_cmd("SET GTO 1")
        self._send_cmd(f"SET SPE {speed}")
        self._send_cmd(f"SET FOR {force}")
        res = self._send_cmd(f"SET POS {pos_int}")
        time.sleep(0.8)
        return res

    def grip_object(self, object_width_m=PROFILE_WIDTH, force=GRIPPER_MAX_FORCE, speed=255):
        """
        物体幅に合わせて強力把持 (常に最大力 force=255)
        ※ 物体を確実に挟み込んでモータトルク把持を効かせるため、
           目標位置は全閉 (POS 255) 方向へ指令し、
           物体に接触した時点で最大把持力 (FOR 255) で確実にホールドさせます。
        """
        self.ensure_activated()
        self._send_cmd("SET GTO 1")
        self._send_cmd(f"SET SPE {speed}")
        self._send_cmd(f"SET FOR {force}")
        res = self._send_cmd("SET POS 255")
        time.sleep(1.0)
        return res

    def get_pos(self):
        """現在のグリッパー開度を取得 (pos_int: 0~255, width_m: 0.0~0.085)"""
        try:
            res = self._send_cmd("GET POS")
            if "POS" in res:
                pos_int = int(res.split()[-1])
                width_m = 0.085 * (1.0 - pos_int / 255.0)
                return pos_int, width_m
            return -1, 0.085
        except Exception:
            return -1, 0.085



class IntegratedPickContactPlace:
    def __init__(self, base, robot_ip=ROBOT_IP):
        self.base = base
        self.robot_ip = robot_ip

        # ロボットモデル初期化
        self.robot, self.gripper = build_ur3_dual_lft()
        self.gripper.set_opening(0.085)  # 初期状態は全開 (85mm)
        self.current_sim_q = VIEW_POSE_Q.copy()
        self.robot.fk(qs=self.current_sim_q)
        self.robot.attach_to(self.base.scene)

        # 実機 RTDE 接続
        self.rtde_c = None
        self.rtde_r = None
        if HAS_UR_RTDE:
            try:
                self.rtde_c = RTDEControlInterface(self.robot_ip)
                self.rtde_r = RTDEReceiveInterface(self.robot_ip)
                print(f"✓ UR3 実機RTDE ({self.robot_ip}) に接続成功！")
            except Exception as e:
                print(f"⚠️ UR3 実機接続失敗 ({e})。シミュレーションモードで実行します。")

        # グリッパー接続
        self.grip_ctrl = RobotiqGripper(self.robot_ip)

        # カメラ設定 & 検出器
        self.cam_yaml = os.path.join(cam_dir, 'camera_params.yaml')
        self.hand_eye_yaml = os.path.join(cam_dir, 'hand_eye_params.yaml')
        self.yaml_save_path = os.path.join(cam_dir, 'detected_profile_pose.yaml')

        self.detector = ProfileDetector(self.cam_yaml, self.hand_eye_yaml)

        self.cap = cv2.VideoCapture(DEVICE_ID)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✓ カメラ (ID: {DEVICE_ID}) 起動成功！")
        else:
            print(f"⚠️ カメラ (ID: {DEVICE_ID}) が見つかりません。")

        # 基準座標系・力覚・公称幾何の初期化 (nrs_vr2準拠)
        self.pos_base_world = np.array([0.0, 0.258485281374, 1.61051471863])
        self.R_base_world = oum.rotmat_from_euler(-3.0 * np.pi / 4.0, 0.0, 0.0)
        self.force_threshold_N = 50.0
        self._load_gravity_calib()

        # 状態変数
        self.detected_obj = None
        self.is_detection_locked = False
        self.is_picked = False
        self.is_gripper_closed = False
        self.is_identifying = False
        self.is_moving = False

        # スレッドセーフなGUI・描画同期待ち変数 (マルチスレッドOpenGL SIGSEGV防止)
        self._pending_status = None
        self._pending_banner = None
        self._pending_gripper_opening = None
        self._pending_robot_q = None

        # プレビュー関連状態変数
        self.preview_action = None       # "VIEW_POSE", "PICK", "CARRY", "PLACE" 等
        self.preview_traj = []           # 補間軌道フレーム一覧
        self.preview_frame_idx = 0
        self.preview_target_q = None

        self.pick_plan_traj = None
        self.place_plan_traj = None
        self.x_identified = None

        # 3Dシーン要素の構築 (ロボット台座、机、座標軸)
        self._setup_3d_scene()

        # HUD 操作ガイドラベルの作成
        self._setup_hud_labels()

        # キーボード入力設定
        self._setup_events()
        self._start_terminal_key_listener()

    def _load_gravity_calib(self):
        self.has_gravity_calib = False
        self.calib_mass = 0.0
        self.calib_f0_tcp = np.zeros(3)
        self.calib_g_base = np.array([0.0, 6.9343, 6.9343])
        calib_file = os.path.join(current_dir, 'tool_gravity_calib.json')
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r', encoding='utf-8') as f:
                    calib = json.load(f)
                self.calib_mass = float(calib.get("mass_kg", 0.0))
                self.calib_f0_tcp = np.array(calib.get("f0_bias_tcp_N", [0.0, 0.0, 0.0]), dtype=float)
                self.calib_g_base = np.array(calib.get("g_base", [0.0, 6.9343, 6.9343]), dtype=float)
                self.has_gravity_calib = True
                print(f"  ✓ 工具重力キャリブレーション読込完了 (質量 m={self.calib_mass:.3f} kg, f0={np.round(self.calib_f0_tcp, 2)})")
            except Exception as e:
                print(f"  ※ キャリブレーション読込失敗 ({e})。生測定値で動作します。")

    def compute_gravity_comp_force(self, R_tcp_base, raw_force_base):
        """
        UR RTDE から取得した raw_force_base (ベース座標系基準の力ベクトル [Fx, Fy, Fz]) に対し、
        工具重力 F_g_base および センサゼロ点オフセット F_bias_base を差し引いた正味外力を算出 (nrs_vr2準拠)
        """
        if not self.has_gravity_calib:
            return raw_force_base
        F_g_base = self.calib_mass * self.calib_g_base
        F_bias_base = R_tcp_base @ self.calib_f0_tcp
        return raw_force_base - F_g_base - F_bias_base

    def get_flange_pose_from_tcp(self, p_tcp_world, R_tcp_world):
        """
        アルミフレーム把持中心 (TCP) の世界座標系位置姿勢から、
        実機 UR コントローラ (ツールフランジ基準) に送信すべき pose (位置 [m] + 回転ベクトル [rad]) を算出 (nrs_vr2準拠)
        """
        p_flange_world = p_tcp_world - R_tcp_world @ TOOL_OFFSET_FLANGE
        p_flange_base = self.R_base_world.T @ (p_flange_world - self.pos_base_world)
        R_flange_base = self.R_base_world.T @ R_tcp_world
        rotvec_base = Rot.from_matrix(R_flange_base).as_rotvec()
        return list(np.concatenate([p_flange_base, rotvec_base]))

    def _setup_3d_scene(self):
        # 1. ワールド原点座標軸 (Frame)
        ossop.frame().attach_to(self.base.scene)

        # 2. ロボット双腕ベース台座 (Base & Frame Mesh)
        base_mesh_path = os.path.join(mesh_dir, 'ur3_dual_base.stl')
        frame_mesh_path = os.path.join(mesh_dir, 'ur3_dual_frame.stl')

        if os.path.exists(base_mesh_path):
            try:
                geom_base = ogl.load_geometry(base_mesh_path)
                base_obj = ossop.mesh(geom_base.vs, geom_base.fs, rgb=ouc.ExtendedColor.DIM_GRAY)
                base_obj.attach_to(self.base.scene)
            except Exception as e:
                print(f"⚠️ ベースメッシュロード警告: {e}")

        if os.path.exists(frame_mesh_path):
            try:
                geom_frame = ogl.load_geometry(frame_mesh_path)
                frame_obj = ossop.mesh(geom_frame.vs, geom_frame.fs, rgb=ouc.ExtendedColor.STEEL_BLUE)
                frame_obj.attach_to(self.base.scene)
            except Exception as e:
                print(f"⚠️ フレームメッシュロード警告: {e}")

        # 3. 作業机 (Table)
        self.table = ossop.box(
            pos=np.array([0.45, 0.0, 1.082], dtype=np.float32),
            xyz_lengths=np.array([0.8, 1.83, 0.024], dtype=np.float32),
            rgb=ouc.ExtendedColor.LIGHT_SLATE_GRAY
        )
        self.table.attach_to(self.base.scene)

        # 机の天板座標フレーム
        ossop.frame(
            pos=np.array([0.45, 0.0, TABLE_Z], dtype=np.float32),
            rotmat=np.eye(3, dtype=np.float32)
        ).attach_to(self.base.scene)

        # 4. 検出物体ボックス (初期状態は非表示)
        self.obj_box = None
        self.contact_spheres = []

    def _setup_hud_labels(self):
        """画面上に操作ガイドおよび状態バナーをオーバーレイ表示 (高コントラスト・深度テスト無効化)"""
        self.hud_card_w, self.hud_card_h = 450, 315
        self.banner_card_w, self.banner_card_h = 620, 38

        card_x = 15
        card_y = self.base.height - self.hud_card_h - 15

        self.hud_bg = shapes.BorderedRectangle(
            x=card_x, y=card_y, width=self.hud_card_w, height=self.hud_card_h,
            border=2, color=(22, 27, 34), border_color=(56, 139, 253)
        )
        self.hud_bg.opacity = 235

        self.hud_title = pyglet.text.Label(
            '【操作ガイド - UR3 アルミフレーム作業】',
            font_name='Noto Sans CJK JP',
            font_size=11,
            weight='bold',
            x=card_x + 15,
            y=card_y + self.hud_card_h - 15,
            anchor_x='left',
            anchor_y='top',
            color=(88, 166, 255, 255)
        )

        self.hud_status = pyglet.text.Label(
            '● 状態: 待機中 (IDLE) - [V]で撮影姿勢プレビュー',
            font_name='Noto Sans CJK JP',
            font_size=9,
            weight='bold',
            x=card_x + 15,
            y=card_y + self.hud_card_h - 38,
            anchor_x='left',
            anchor_y='top',
            color=(63, 185, 80, 255)
        )

        guide_text = (
            "  [V]   : 撮影待機 (View Pose) プレビュー\n"
            "  [S]   : カメラ認識確定 ＆ ピック計画プレビュー\n"
            "  [1]   : ピック動作プレビュー ➔ [1]で実機実行\n"
            "  [2]   : 搬送動作プレビュー ➔ [2]で実機実行\n"
            "  [3]   : 3回接触同定プレビュー ➔ [3]で実機実行\n"
            "  [4]   : 精密設置プレビュー ➔ [4]で実機実行\n"
            "  [Enter]/[Y] : プレビュー中の動作を実機へ送信実行！\n"
            "  [ESC] : プレビューキャンセル / 中断\n"
            "  [H]   : ホーム復帰 | [O]/[C]: グリッパ手動開/閉\n"
            "--------------------------------------------------\n"
            "【マウス操作】\n"
            "  右ドラッグ: 視点回転 | 中ドラッグ: 視点移動 | ホイール: ズーム"
        )
        self.hud_body = pyglet.text.Label(
            guide_text,
            font_name='Noto Sans CJK JP',
            font_size=9,
            x=card_x + 15,
            y=card_y + self.hud_card_h - 60,
            anchor_x='left',
            anchor_y='top',
            multiline=True,
            width=self.hud_card_w - 30,
            color=(230, 237, 243, 255)
        )

        b_x = (self.base.width - self.banner_card_w) // 2
        b_y = self.base.height - self.banner_card_h - 15
        self.banner_bg = shapes.BorderedRectangle(
            x=b_x, y=b_y, width=self.banner_card_w, height=self.banner_card_h,
            border=2, color=(24, 32, 24), border_color=(46, 160, 67)
        )
        self.banner_bg.opacity = 240

        self.banner_label = pyglet.text.Label(
            '✓ 準備完了: [V] キーを押して撮影姿勢プレビューを開始できます',
            font_name='Noto Sans CJK JP',
            font_size=11,
            weight='bold',
            x=self.base.width // 2,
            y=b_y + self.banner_card_h // 2,
            anchor_x='center',
            anchor_y='center',
            color=(86, 211, 100, 255)
        )

        # ウィンドウサイズ変更時に HUD 位置を追従
        orig_on_resize = self.base.on_resize

        def custom_on_resize(width, height):
            orig_on_resize(width, height)
            self._update_hud_positions(width, height)

        self.base.on_resize = custom_on_resize

        # 画面描画フック (2D HUD は深度テスト無効化で最前面描画)
        orig_on_draw = self.base.on_draw

        def custom_on_draw():
            orig_on_draw()

            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_FALSE)

            if hasattr(self, 'hud_bg'):
                self.hud_bg.draw()
            if hasattr(self, 'hud_title'):
                self.hud_title.draw()
            if hasattr(self, 'hud_status'):
                self.hud_status.draw()
            if hasattr(self, 'hud_body'):
                self.hud_body.draw()

            if hasattr(self, 'banner_label') and self.banner_label.text:
                if hasattr(self, 'banner_bg'):
                    self.banner_bg.draw()
                self.banner_label.draw()

            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)

        self.base.on_draw = custom_on_draw

    def _update_hud_positions(self, width, height):
        """ウィンドウサイズに合わせて HUD とバナーの位置を自動再計算"""
        if hasattr(self, 'hud_bg'):
            card_x = 15
            card_y = height - self.hud_card_h - 15
            self.hud_bg.x = card_x
            self.hud_bg.y = card_y
            self.hud_title.x = card_x + 15
            self.hud_title.y = card_y + self.hud_card_h - 15
            self.hud_status.x = card_x + 15
            self.hud_status.y = card_y + self.hud_card_h - 38
            self.hud_body.x = card_x + 15
            self.hud_body.y = card_y + self.hud_card_h - 60

        if hasattr(self, 'banner_bg'):
            b_x = (width - self.banner_card_w) // 2
            b_y = height - self.banner_card_h - 15
            self.banner_bg.x = b_x
            self.banner_bg.y = b_y
            self.banner_label.x = width // 2
            self.banner_label.y = b_y + self.banner_card_h // 2

    def _unwrap_to_prev(self, target_q, prev_q):
        return _unwrap_to_prev(target_q, prev_q)

    def set_status(self, text, color=(63, 185, 80, 255)):
        """HUD の状態表示を更新 (スレッドセーフ: メインスレッドの update() で安全に適用)"""
        self._pending_status = (text, color)

    def set_banner(self, text, border_color=(46, 160, 67), text_color=(86, 211, 100, 255), bg_color=(24, 32, 24)):
        """画面上部バナーメッセージを更新 (スレッドセーフ: メインスレッドの update() で安全に適用)"""
        self._pending_banner = (text, border_color, text_color, bg_color)

    def set_gripper_opening_safe(self, opening_m):
        """シミュレーション上のグリッパー開度を更新 (スレッドセーフ: メインスレッドの update() で安全に適用)"""
        self._pending_gripper_opening = opening_m

    def set_robot_q_safe(self, q):
        """シミュレーション上のロボット関節角を更新 (スレッドセーフ: メインスレッドの update() で安全に適用)"""
        self._pending_robot_q = np.array(q).copy()

    def handle_key_input(self, key_id):
        """カメラ画面(OpenCV)、3D画面(Pyglet)、端末(stdin)のいずれからキー入力があっても一元的に即時処理"""
        c = None
        if isinstance(key_id, str):
            c = key_id.lower()
        elif isinstance(key_id, int):
            if key_id == 27:
                c = '\x1b'
            elif key_id in (10, 13):
                c = '\n'
            elif key_id == 32:
                c = ' '
            elif 0 <= key_id <= 255:
                c = chr(key_id).lower()

        if c is None:
            return

        print(f"[キー入力検知]: {repr(c)} (preview_action={self.preview_action})")

        # 1. プレビュー確認 ＆ 実行 (Enter, Y, スペース)
        if c in ['\n', '\r', 'y', ' ']:
            self.on_confirm_preview()
        elif c == '1':
            if self.preview_action == "PICK_AND_CARRY":
                self.on_confirm_preview()
            else:
                self.on_key_1_pick_down()
        elif c == '2':
            if self.preview_action == "LIFT_AND_CARRY":
                self.on_confirm_preview()
            else:
                self.on_key_2_lift_and_carry()
        elif c == '3':
            if self.preview_action == "CONTACT_IDENTIFY":
                self.on_confirm_preview()
            else:
                self.on_key_3_contact_identify()
        elif c == '4':
            if self.preview_action == "EXECUTE_PLACE":
                self.on_confirm_preview()
            else:
                self.on_key_4_execute_place()

        # 3. 撮影待機姿勢 (V)
        elif c == 'v':
            self.request_move_to_view_pose()

        # 4. 検出姿勢確定 ＆ ピック動作計画 (S, P)
        elif c in ['s', 'p']:
            self.on_save_detection()

        # 5. ホーム姿勢復帰 (H)
        elif c == 'h':
            self.on_return_home()

        # 6. グリッパー開閉 (O: 開, C: 閉)
        elif c == 'o':
            self.on_open_gripper()
        elif c == 'c':
            self.on_close_gripper()

        # 7. キャンセル / 非常停止 (ESC, Q)
        elif c in ['\x1b', 'q']:
            self.on_abort()

    def _setup_events(self):
        @self.base.event
        def on_key_press(symbol, modifiers):
            # テンキーを含めて全キーに対応
            num_1 = getattr(key, 'NUM_1', key._1)
            num_2 = getattr(key, 'NUM_2', key._2)
            num_3 = getattr(key, 'NUM_3', key._3)
            num_4 = getattr(key, 'NUM_4', key._4)
            num_enter = getattr(key, 'NUM_ENTER', key.ENTER)

            if symbol == key.V:
                self.handle_key_input('v')
            elif symbol in [key.ENTER, num_enter, key.Y, key.SPACE]:
                self.handle_key_input('\n')
            elif symbol in [key._1, num_1]:
                self.handle_key_input('1')
            elif symbol in [key._2, num_2]:
                self.handle_key_input('2')
            elif symbol in [key._3, num_3]:
                self.handle_key_input('3')
            elif symbol in [key._4, num_4]:
                self.handle_key_input('4')
            elif symbol in [key.S, key.P]:
                self.handle_key_input('s')
            elif symbol == key.H:
                self.handle_key_input('h')
            elif symbol == key.O:
                self.handle_key_input('o')
            elif symbol == key.C:
                self.handle_key_input('c')
            elif symbol in [key.ESCAPE, key.Q]:
                self.handle_key_input('\x1b')

    def _start_terminal_key_listener(self):
        def listener():
            try:
                if not sys.stdin.isatty():
                    return
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while True:
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if rlist:
                            ch = sys.stdin.read(1)
                            self.handle_key_input(ch)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

        t = threading.Thread(target=listener, daemon=True)
        t.start()

    # ===========================================================================
    # 撮影待機姿勢 (View Pose) プレビュー＆確認実行
    # ===========================================================================
    def request_move_to_view_pose(self):
        """[V] キー: まずシミュレーションで撮影待機姿勢への軌道をプレビュー再生"""
        if self.is_moving:
            print("⚠️ ロボットが動作中のため受け付けられません。")
            return

        # 現在の関節角 (実機があれば実機角、なければシミュレーション現在角)
        if self.rtde_r is not None:
            try:
                start_q = np.array(self.rtde_r.getActualQ())
            except Exception:
                start_q = self.current_sim_q.copy()
        else:
            start_q = self.current_sim_q.copy()

        goal_q = VIEW_POSE_Q.copy()

        # 40フレームのスムーズ補間軌道 (Cosine イージング)
        n_steps = 45
        traj = []
        for i in range(n_steps + 1):
            s = (1.0 - np.cos(np.pi * (i / n_steps))) / 2.0
            q_interp = start_q + s * (goal_q - start_q)
            traj.append(q_interp)

        self.preview_action = "VIEW_POSE"
        self.preview_traj = traj
        self.preview_frame_idx = 0
        self.preview_target_q = goal_q

        print("\n" + "=" * 70)
        print(" 【シミュレーション確認中: 撮影待機姿勢 (View Pose)】")
        print("   画面上でロボットが待機姿勢へ移動する動きを確認してください。")
        print("   ・[Enter] または [Y] キー : 確認完了 ➔ 実機へ送信して動作実行！")
        print("   ・[ESC] キー              : キャンセル（実機は動かしません）")
        print("=" * 70 + "\n")

    def on_confirm_preview(self):
        """[Enter] / [Y] / 各数字キー: シミュレーション確認完了 ➔ 実機実行"""
        act = self.preview_action
        if act == "VIEW_POSE":
            target_q = self.preview_target_q
            self.preview_action = None
            self.preview_traj = []
            threading.Thread(target=self._worker_execute_view_pose, args=(target_q,), daemon=True).start()
        elif act == "PICK_AND_CARRY":
            self.preview_action = None
            self.preview_traj = []
            threading.Thread(target=self._worker_execute_pick, daemon=True).start()
        elif act == "LIFT_AND_CARRY":
            self.preview_action = None
            self.preview_traj = []
            threading.Thread(target=self._worker_execute_lift_and_carry, daemon=True).start()
        elif act == "CONTACT_IDENTIFY":
            self.preview_action = None
            self.preview_traj = []
            threading.Thread(target=self._worker_execute_contact_and_identify, daemon=True).start()
        elif act == "EXECUTE_PLACE":
            self.preview_action = None
            self.preview_traj = []
            threading.Thread(target=self._worker_execute_place, daemon=True).start()
        elif act is None:
            if self.pick_plan_traj is not None and not self.is_picked:
                self.on_key_1_pick_down()
            else:
                print("※ 現在プレビュー待機中の動作はありません。")

    def _worker_execute_view_pose(self, target_q):
        self.is_moving = True
        try:
            print("\n>>> 実機へ送信: 撮影待機姿勢 (View Pose) へ移動中...")
            if self.rtde_c is not None:
                self.rtde_c.moveJ(target_q, SPEED_NORMAL, ACCEL_NORMAL)
                time.sleep(1.0)
                if self.rtde_r is not None:
                    cur_q = np.array(self.rtde_r.getActualQ())
                    self.robot.fk(qs=cur_q)
                    self.current_sim_q = cur_q.copy()
            else:
                self.robot.fk(qs=target_q)
                self.current_sim_q = target_q.copy()

            self.is_detection_locked = False
            print("✓ 実機が撮影待機姿勢に到着しました！カメラ検出を開始します。")
        finally:
            self.is_moving = False

    # ===========================================================================
    # メイン更新ループ
    # ===========================================================================
    def update(self, dt):
        """定期描画・アニメーション・カメラ更新ループ (メインスレッドで実行)"""

        # A. 保留中の HUD / バナー更新をメインスレッドで安全に適用 (SIGSEGV防止)
        if self._pending_status is not None:
            text, color = self._pending_status
            self._pending_status = None
            if hasattr(self, 'hud_status'):
                self.hud_status.text = text
                self.hud_status.color = color

        if self._pending_banner is not None:
            text, border_color, text_color, bg_color = self._pending_banner
            self._pending_banner = None
            if hasattr(self, 'banner_label'):
                self.banner_label.text = text
                self.banner_label.color = text_color
            if hasattr(self, 'banner_bg'):
                self.banner_bg.border_color = border_color
                self.banner_bg.color = bg_color

        # B. 保留中のロボット・グリッパー更新をメインスレッドで安全に適用 (SIGSEGV防止)
        if self._pending_gripper_opening is not None:
            op = self._pending_gripper_opening
            self._pending_gripper_opening = None
            self.gripper.set_opening(op)

        if self._pending_robot_q is not None:
            q_target = self._pending_robot_q
            self._pending_robot_q = None
            self.robot.fk(qs=q_target)
            self.current_sim_q = q_target.copy()

        # 1. プレビュー再生中の場合: アニメーションを進める
        if self.preview_action is not None and len(self.preview_traj) > 0:
            if self.preview_frame_idx < len(self.preview_traj):
                frame_item = self.preview_traj[self.preview_frame_idx]
                if isinstance(frame_item, tuple):
                    q_frame, jaw_w, is_gripped = frame_item
                else:
                    q_frame, jaw_w, is_gripped = frame_item, 0.085, False

                self.robot.fk(qs=q_frame)
                self.gripper.set_opening(jaw_w)
                self.current_sim_q = q_frame.copy()

                # ★ プレビュー中の物体追従描画 (把持中は常に手先に追従して3D描画)
                if self.pick_plan_traj is not None:
                    if is_gripped:
                        tcp = self.gripper.tcp('grasp_center')
                        tcp_pos = tcp.tf[:3, 3]
                        tcp_rot = tcp.tf[:3, :3]
                        cur_p = tcp_pos + tcp_rot @ self.pick_plan_traj['p_rel']
                        cur_R = tcp_rot @ self.pick_plan_traj['R_rel']
                        self._render_profile_box_direct(cur_p, cur_R)
                    elif self.preview_action == "PICK_AND_CARRY":
                        p_center = self.pick_plan_traj['p_center']
                        R_obj_orig = self.pick_plan_traj['R_obj_orig']
                        self._render_profile_box_direct(p_center, R_obj_orig)

                self.preview_frame_idx += 1
            else:
                # プレビュー動作をスムーズにループ再生
                self.preview_frame_idx = 0

        # 2. 実機追従 (プレビュー中でない場合)
        elif self.preview_action is None:
            if self.rtde_r is not None and not self.is_moving:
                try:
                    cur_q = np.array(self.rtde_r.getActualQ())
                    self.robot.fk(qs=cur_q)
                    self.current_sim_q = cur_q.copy()
                except Exception:
                    pass

            if self.is_picked:
                self.gripper.set_opening(PROFILE_WIDTH)
                if self.pick_plan_traj is not None:
                    tcp = self.gripper.tcp('grasp_center')
                    tcp_pos = tcp.tf[:3, 3]
                    tcp_rot = tcp.tf[:3, :3]
                    cur_p = tcp_pos + tcp_rot @ self.pick_plan_traj['p_rel']
                    cur_R = tcp_rot @ self.pick_plan_traj['R_rel']
                    self._render_profile_box_direct(cur_p, cur_R)
            elif not self.is_moving:
                self.gripper.set_opening(0.085)

        # 3. カメラ画像取得 & リアルタイム描画 & キー入力処理 (毎フレーム必ず実行)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                display = frame.copy()

                # プレビュー中、確定中、またはロボット動作中でない場合のみリアルタイム物体検出を更新
                if not self.is_detection_locked and self.preview_action is None and not self.is_moving:
                    flange_tf = self.robot.tcp('flange').tf
                    T_w_cam = flange_tf @ self.detector.T_c2g
                    cam_pos_w = T_w_cam[:3, 3]
                    cam_rot_w = T_w_cam[:3, :3]

                    obj_info, _ = self.detector.detect_in_frame(frame, cam_pos_w, cam_rot_w)
                    if obj_info is not None:
                        self.detected_obj = obj_info
                        cv2.drawContours(display, [obj_info['box_pts']], 0, (0, 255, 0), 2)
                        cx, cy = int(obj_info['pixel_center'][0]), int(obj_info['pixel_center'][1])
                        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                        pw = obj_info['pos_world']
                        cv2.putText(display, f"XYZ: [{pw[0]:.3f}, {pw[1]:.3f}, {pw[2]:.3f}] m",
                                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        cv2.putText(display, f"Yaw: {obj_info['yaw_deg']:+.1f} deg",
                                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                        self._render_profile_box(pw, obj_info['yaw_deg'])
                elif self.preview_action == "PICK_AND_CARRY":
                    cv2.putText(display, "PREVIEWING PICK & CARRY", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    cv2.putText(display, "[1] or [Enter]: Execute Pick", (15, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(display, "[ESC]: Cancel Preview", (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                elif self.preview_action == "LIFT_AND_CARRY":
                    cv2.putText(display, "PREVIEWING LIFT & CARRY", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    cv2.putText(display, "[2] or [Enter]: Execute Carry", (15, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(display, "[ESC]: Cancel Preview", (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                elif self.preview_action == "CONTACT_IDENTIFY":
                    cv2.putText(display, "PREVIEWING 3-CONTACT SEQUENCE", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    cv2.putText(display, "[3] or [Enter]: Execute Contact ID", (15, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(display, "[ESC]: Cancel Preview", (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                elif self.preview_action == "EXECUTE_PLACE":
                    cv2.putText(display, "PREVIEWING PRECISION PLACE", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    cv2.putText(display, "[4] or [Enter]: Execute Place", (15, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(display, "[ESC]: Cancel Preview", (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                elif self.is_detection_locked:
                    cv2.putText(display, "DETECTION LOCKED", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Camera View (one)", display)

            # ★ 毎フレーム必ず waitKey(1) を呼び出して OpenCV イベントキューを処理
            cv_key = cv2.waitKey(1) & 0xFF
            if cv_key != 255:
                self.handle_key_input(cv_key)

    def _render_profile_box(self, pos, yaw_deg):
        rotmat = oum.rotmat_from_euler(0.0, 0.0, np.radians(yaw_deg))
        center_pos = pos.copy()
        center_pos[2] -= PROFILE_HEIGHT / 2.0
        self._render_profile_box_direct(center_pos, rotmat)

    def _render_profile_box_direct(self, center_pos, rotmat):
        """メッシュの毎フレーム再生成(detach/attach)を廃止し、位置姿勢行列の直接更新でSIGSEGVを完全に防止"""
        if self.obj_box is None:
            self.obj_box = ossop.box(
                pos=center_pos,
                xyz_lengths=np.array([PROFILE_LENGTH, PROFILE_WIDTH, PROFILE_HEIGHT], dtype=np.float32),
                rotmat=rotmat,
                rgb=ouc.BasicColor.BLUE,
                alpha=0.8
            )
            self.obj_box.attach_to(self.base.scene)
        else:
            self.obj_box.set_pos_rotmat(pos=center_pos, rotmat=rotmat)

    def on_save_detection(self):
        if self.is_moving:
            print("⚠️ ロボット動作中のため受け付けられません。")
            return

        # プレビュー中ならプレビューを解除して再計画
        if self.preview_action is not None:
            self.preview_action = None
            self.preview_traj = []

        if self.detected_obj is None:
            # もしリアルタイム検出がフレーム抜けしていた場合、前回の保存ファイルから読み込みを試行
            if os.path.exists(self.yaml_save_path):
                try:
                    with open(self.yaml_save_path, 'r') as f:
                        saved = yaml.safe_load(f)
                    self.detected_obj = {
                        'pos_world': np.array(saved['pos_world']),
                        'yaw_deg': saved['yaw_deg']
                    }
                    print(f"✓ 前回の検出姿勢ファイル ({self.yaml_save_path}) を再利用します。")
                except Exception:
                    pass

        if self.detected_obj is None:
            print("⚠️ 物体がまだ検出されていません。カメラ視野内にアルミフレームを置いてください。")
            self.set_banner("⚠️ 物体が検出されていません。カメラ視野内にアルミフレームを置いてください",
                            border_color=(218, 54, 51), text_color=(255, 120, 120, 255), bg_color=(40, 20, 20))
            return

        self.is_detection_locked = True
        pw = self.detected_obj['pos_world']
        yaw = self.detected_obj['yaw_deg']
        print(f"\n★ 検出姿勢を確定・保存しました: XYZ=[{pw[0]:.3f}, {pw[1]:.3f}, {pw[2]:.3f}], Yaw={yaw:+.1f} deg")

        data = {
            'pos_world': [float(v) for v in pw],
            'yaw_deg': float(yaw),
        }
        with open(self.yaml_save_path, 'w') as f:
            yaml.dump(data, f)

        # 3Dシーンの直方体ボックスを確定姿勢で更新
        self._render_profile_box(pw, yaw)

        # 動作計画＆シミュレーションプレビュー開始
        self.plan_pick()

    def plan_pick(self):
        """検出位置に基づくピック動作計画 ＆ シミュレーションプレビュー生成"""
        if self.detected_obj is None:
            print("⚠️ 物体検出情報がありません。先に [S] キー等で確定してください。")
            return

        pw = self.detected_obj['pos_world']
        obj_yaw = self.detected_obj['yaw_deg']

        tcp = self.gripper.tcp('grasp_center')

        # ★ 短辺 (幅 60mm) を挟む2つの候補把持角度 (+/- 90度)
        # アルミフレームは長さ 160mm, 幅 60mm。
        # グリッパーの開口幅は 85mm なので、長軸方向は把持不可能。
        # グリッパーの開閉軸 (ローカル X 軸) を短軸方向に向けるため、長軸から ±90度回転させる。
        cand_yaws = [
            (obj_yaw + 90.0 + 180.0) % 360.0 - 180.0,
            (obj_yaw - 90.0 + 180.0) % 360.0 - 180.0
        ]

        best_plan = None
        min_dist = float('inf')

        for gyaw in cand_yaws:
            R_yaw = oum.rotmat_from_euler(0.0, 0.0, np.radians(gyaw))
            R_target = R_yaw @ R_BASE  # 鉛直真下トップダウン把持姿勢

            # ★ 幾何学的整合性: グリッパの +Y_tcp 軸が SAFE_HOME_Q で鉛直真下を向く
            # したがって、中心から -Y_tcp 方向へ 35mm オフセットした位置を把持
            # これにより、把持点から底面までの距離が厳密に 115mm (上面から 45mm 潜った深さ) となり、
            # CARRY_HOME_Q に搬送された際にアルミフレーム下端が鉛直真下を向き、机面上 11.5cm 余裕が確保される
            y_tcp = R_target[:, 1]
            p_g_xy = pw[:2] - GRASP_CENTER_OFFSET_M * y_tcp[:2]
            p_grasp = np.array([p_g_xy[0], p_g_xy[1], TABLE_Z + 0.030])
            p_standby = np.array([p_grasp[0], p_grasp[1], TABLE_Z + 0.180])
            p_lift = np.array([p_grasp[0], p_grasp[1], TABLE_Z + 0.220])

            sols_g = self.robot.ik(p_grasp, R_target, tcp=tcp)
            if not sols_g:
                continue

            for s in sols_g:
                su = _unwrap_to_prev(s, SAFE_HOME_Q)

                # standby IK
                sols_sb = self.robot.ik(p_standby, R_target, tcp=tcp)
                best_sb = None
                min_dsb = float('inf')
                for sb in sols_sb:
                    sbu = _unwrap_to_prev(sb, su)
                    dsb = np.linalg.norm(sbu - su)
                    if dsb < min_dsb:
                        min_dsb = dsb
                        best_sb = sbu

                # lift IK
                sols_l = self.robot.ik(p_lift, R_target, tcp=tcp)
                best_lift = None
                min_dl = float('inf')
                for sl in sols_l:
                    slu = _unwrap_to_prev(sl, su)
                    dl = np.linalg.norm(slu - su)
                    if dl < min_dl:
                        min_dl = dl
                        best_lift = slu

                if best_sb is not None and best_lift is not None:
                    q_lift = _unwrap_to_prev(best_lift, SAFE_HOME_Q)
                    su = _unwrap_to_prev(su, q_lift)
                    q_sb = _unwrap_to_prev(best_sb, su)

                    # 遷移距離の合計スコア (VIEW_POSE_Q から把持、把持からホームへの関節変化量)
                    d = np.linalg.norm(su - VIEW_POSE_Q) + np.linalg.norm(q_lift - SAFE_HOME_Q)
                    if d < min_dist:
                        min_dist = d
                        best_plan = {
                            'gyaw': gyaw,
                            'p_grasp': p_grasp,
                            'p_standby': p_standby,
                            'p_lift': p_lift,
                            'R_target': R_target,
                            'approach_q': q_sb,
                            'grip_q': su,
                            'lift_q': q_lift,
                            'carry_q': _unwrap_to_prev(SAFE_HOME_Q, q_lift)
                        }

        if best_plan is None:
            print("❌ ピック動作の逆運動学 (IK) が解けませんでした。位置・角度を確認してください。")
            self.set_banner("❌ ピック動作のIKが解けませんでした。位置や角度を確認してください",
                            border_color=(218, 54, 51), text_color=(255, 120, 120, 255), bg_color=(40, 20, 20))
            self.pick_plan_traj = None
            return

        # 把持時での物体との相対位置関係を計算
        p_center = pw.copy()
        p_center[2] -= PROFILE_HEIGHT / 2.0
        R_obj_orig = oum.rotmat_from_euler(0.0, 0.0, np.radians(obj_yaw))

        self.robot.fk(qs=best_plan['grip_q'])
        tcp_grip_pos = tcp.tf[:3, 3]
        tcp_grip_rot = tcp.tf[:3, :3]
        p_rel = tcp_grip_rot.T @ (p_center - tcp_grip_pos)
        R_rel = tcp_grip_rot.T @ R_obj_orig

        best_plan['p_center'] = p_center
        best_plan['R_obj_orig'] = R_obj_orig
        best_plan['p_rel'] = p_rel
        best_plan['R_rel'] = R_rel

        self.pick_plan_traj = best_plan
        q_app = best_plan['approach_q']
        q_grip = best_plan['grip_q']
        q_lift = best_plan['lift_q']
        q_carry = best_plan['carry_q']

        print("\n" + "=" * 70)
        print("✓ ピック＆搬送動作計画が完了しました！")
        print(f"  ・アルミフレーム長軸 : {obj_yaw:+.1f}°")
        print(f"  ・グリッパ開閉把持軸 : {best_plan['gyaw']:+.1f}° (短辺 60mm を挟む向き)")
        print(f"  ・把持目標中心 : [{best_plan['p_grasp'][0]:.4f}, {best_plan['p_grasp'][1]:.4f}, {best_plan['p_grasp'][2]:.4f}] m")
        print("  ・シミュレーションで動作（アプローチ➔把持➔上昇➔3回接触ホーム搬送）をプレビュー中。")
        print("  ・[1] または [Enter] / [Y] : 実機が直上アプローチ ➔ 把持深さへ下降して停止します！")
        print("  ・（★ 下降停止後、[C] キーで手動把持してください。勝手には閉じません）")
        print("  ・[ESC] : キャンセル")
        print("=" * 70 + "\n")

        # プレビュー用マルチフェーズ補間軌道 (全 92 フレーム)
        # 現在姿勢 -> アプローチ -> 下降 -> 物体幅把持 -> リフト -> ホーム搬送
        start_q = self.current_sim_q.copy()
        traj = []
        # Phase 1: start -> q_app (25 frames, 全開 85mm)
        for i in range(25):
            s = (1.0 - np.cos(np.pi * (i / 24))) / 2.0
            traj.append((start_q + s * (q_app - start_q), 0.085, False))
        # Phase 2: q_app -> q_grip (15 frames, 全開 85mmで物体を跨ぐ)
        for i in range(15):
            s = (1.0 - np.cos(np.pi * (i / 14))) / 2.0
            traj.append((q_app + s * (q_grip - q_app), 0.085, False))
        # Phase 3: 把持 (12 frames, 爪幅 85mm -> 物体幅 60mm に滑らかに閉じる)
        for i in range(12):
            u = i / 11.0
            w = (1.0 - u) * 0.085 + u * PROFILE_WIDTH
            traj.append((q_grip.copy(), w, (u >= 0.5)))
        # Phase 4: q_grip -> q_lift (15 frames, 物体幅 60mm で挟んだまま上昇)
        for i in range(15):
            s = (1.0 - np.cos(np.pi * (i / 14))) / 2.0
            traj.append((q_grip + s * (q_lift - q_grip), PROFILE_WIDTH, True))
        # Phase 5: q_lift -> q_carry (25 frames, 物体幅 60mm で把持維持のまま3回接触ホームへ搬送)
        for i in range(25):
            s = (1.0 - np.cos(np.pi * (i / 24))) / 2.0
            traj.append((q_lift + s * (q_carry - q_lift), PROFILE_WIDTH, True))

        self.preview_action = "PICK_AND_CARRY"
        self.preview_traj = traj
        self.preview_frame_idx = 0

        self.set_status("● 状態: ピック計画プレビュー中 - [1]または[Enter]で実機実行", color=(255, 215, 0, 255))
        self.set_banner("★【ピック計画プレビュー中】 [1] または [Enter] で下降停止！ (把持は [C] キー)",
                        border_color=(230, 180, 40), text_color=(255, 230, 80, 255), bg_color=(36, 32, 20))
        print("=" * 70 + "\n")

    def on_key_1_pick_down(self):
        if self.is_moving:
            return
        if self.pick_plan_traj is None:
            print("⚠️ 先に [S] キーで検出姿勢を確定し、動作計画を行ってください。")
            return
        threading.Thread(target=self._worker_execute_pick, daemon=True).start()

    def _worker_execute_pick(self):
        self.is_moving = True
        try:
            print("\n" + "=" * 70)
            print("【★★★ 実機動作 1: ピックアプローチ＆把持深さへ下降 開始 ★★★】")
            print("  1. グリッパ全開 (85mm) に開放")
            print("  2. 直上アプローチ位置へ安全移動")
            print("  3. 把持深さへ慎重下降して【停止】 (把持は勝手に行いません)")
            print("=" * 70 + "\n")

            print(">> 1/3: グリッパ全開 (85mm) に開放中...")
            self.set_gripper_opening_safe(0.085)
            self.grip_ctrl.open()
            time.sleep(0.5)

            if self.rtde_c is not None:
                print(f">> 2/3: アプローチ位置へ安全移動中 (速度: {SPEED_NORMAL} rad/s)...")
                self.rtde_c.moveJ(self.pick_plan_traj['approach_q'], SPEED_NORMAL, ACCEL_NORMAL)
                time.sleep(0.2)
                print(f">> 3/3: 把持深さへ慎重下降中 (速度: {SPEED_SLOW} rad/s)...")
                self.rtde_c.moveJ(self.pick_plan_traj['grip_q'], SPEED_SLOW, ACCEL_SLOW)
                time.sleep(0.3)
                if self.rtde_r is not None:
                    cur_q = np.array(self.rtde_r.getActualQ())
                    self.robot.fk(qs=cur_q)
                    self.current_sim_q = cur_q.copy()
            else:
                self.robot.fk(qs=self.pick_plan_traj['grip_q'])
                self.current_sim_q = self.pick_plan_traj['grip_q'].copy()

            self.is_picked = False
            self.is_gripper_closed = False

            self.set_status("● 状態: 把持位置到達・停止中 - [C]で把持", color=(255, 215, 0, 255))
            self.set_banner("✓ 把持位置に到達・停止しました！ [C] キーを押して把持してください ➔ [2] で搬送",
                            border_color=(230, 180, 40), text_color=(255, 230, 80, 255), bg_color=(36, 32, 20))

            print("\n" + "=" * 70)
            print("✓ 【把持位置に到達・停止しました！】")
            print("  ・[C] キーを押してグリッパを閉じ、アルミフレームを把持してください。")
            print("  ・微調整が必要なら [O] でグリッパを開くことも可能です。")
            print("  ★ 把持が完了したら [2] キーを押してリフト＆ホーム搬送を実行してください！")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"⚠️ ピック下降動作中にエラーが発生しました: {e}")
        finally:
            self.is_moving = False

    def on_key_2_lift_and_carry(self):
        """[2] キー: 搬送動作プレビュー開始 ➔ [2] または [Enter] で実機実行"""
        if self.is_moving:
            print("⚠️ ロボット移動中のため受け付けられません。")
            return
        if not self.is_picked:
            print("⚠️ 物体をまだ把持していません。先に [C] キーを押してワークを把持してください。")
            self.set_banner("⚠️ 物体を把持していません！ 先に [C] キーで把持してください",
                            border_color=(200, 60, 60), text_color=(255, 100, 100, 255), bg_color=(40, 20, 20))
            return

        if self.preview_action == "LIFT_AND_CARRY":
            self.on_confirm_preview()
            return

        start_q = self.current_sim_q.copy()
        if self.rtde_r is not None:
            try:
                start_q = np.array(self.rtde_r.getActualQ())
            except Exception:
                pass

        q_l = self.pick_plan_traj.get('lift_q') if self.pick_plan_traj is not None else None
        if q_l is None:
            self.robot.fk(qs=start_q)
            tcp = self.gripper.tcp('grasp_center')
            lift_pos = tcp.tf[:3, 3] + np.array([0.0, 0.0, 0.15])
            sols = self.robot.ik(lift_pos, tcp.tf[:3, :3], tcp=tcp)
            q_l = pick_closest_ik(sols, start_q)
            if q_l is None:
                q_l = start_q

        q_c = self.pick_plan_traj.get('carry_q') if self.pick_plan_traj is not None else CARRY_HOME_Q

        traj = []
        # Phase 1: start_q -> q_l (15 frames)
        for i in range(15):
            s = (1.0 - np.cos(np.pi * (i / 14))) / 2.0
            traj.append((start_q + s * (q_l - start_q), PROFILE_WIDTH, True))

        # Phase 2: q_l -> q_c (25 frames)
        for i in range(25):
            s = (1.0 - np.cos(np.pi * (i / 24))) / 2.0
            traj.append((q_l + s * (q_c - q_l), PROFILE_WIDTH, True))

        # Phase 3: carry 待機 (10 frames)
        for _ in range(10):
            traj.append((q_c.copy(), PROFILE_WIDTH, True))

        self.preview_action = "LIFT_AND_CARRY"
        self.preview_traj = traj
        self.preview_frame_idx = 0

        self.set_status("● 状態: 搬送計画プレビュー中 - [2]または[Enter]で実機実行", color=(255, 215, 0, 255))
        self.set_banner("★【搬送動作プレビュー中】 [2] または [Enter] で実機へ送信実行！ | [ESC] でキャンセル",
                        border_color=(230, 180, 40), text_color=(255, 230, 80, 255), bg_color=(36, 32, 20))

        print("\n" + "=" * 70)
        print("✓ 持ち上げ＆接触ホーム搬送のシミュレーションプレビューを開始しました！")
        print("  ・目標姿勢 : 3回接触ホーム (直立・机面上 11.5cm 待機)")
        print("  ・[2] または [Enter] / [Y] : 実機へ送信して搬送を実行！")
        print("  ・[ESC] : キャンセル")
        print("=" * 70 + "\n")

    def _worker_execute_lift_and_carry(self):
        self.is_moving = True
        try:
            print("\n>>> [2] 持ち上げ・接触ホーム姿勢への搬送を開始します...")
            q_l = self.pick_plan_traj.get('lift_q') if self.pick_plan_traj is not None else None
            q_c = self.pick_plan_traj.get('carry_q') if self.pick_plan_traj is not None else CARRY_HOME_Q

            if self.rtde_c is not None:
                if q_l is not None:
                    print(f"  ・計画済みリフト位置へ鉛直上昇中 (速度: {SPEED_SLOW} rad/s)...")
                    self.rtde_c.moveJ(q_l, SPEED_SLOW, ACCEL_SLOW)
                else:
                    cur_q = np.array(self.rtde_r.getActualQ())
                    self.robot.fk(qs=cur_q)
                    tcp = self.gripper.tcp('grasp_center')
                    lift_pos = tcp.tf[:3, 3] + np.array([0.0, 0.0, 0.15])
                    sols = self.robot.ik(lift_pos, tcp.tf[:3, :3], tcp=tcp)
                    q_lift = pick_closest_ik(sols, cur_q)
                    if q_lift is not None:
                        self.rtde_c.moveJ(q_lift, SPEED_SLOW, ACCEL_SLOW)

                print(f"  ・3回接触ホーム姿勢へ安全搬送中 (速度: {SPEED_NORMAL} rad/s)...")
                self.rtde_c.moveJ(q_c, SPEED_NORMAL, ACCEL_NORMAL)
            else:
                self.robot.fk(qs=q_c)
                self.current_sim_q = q_c.copy()

            self.set_status("● 状態: 搬送完了 - [3]で3回接触同定開始", color=(63, 185, 80, 255))
            self.set_banner("✓ 搬送完了！ [3] キーを押して3回接触同定を開始してください",
                            border_color=(46, 160, 67), text_color=(86, 211, 100, 255), bg_color=(24, 32, 24))
            print("✓ [2] 搬送完了！ [3] キーを押して3回接触同定を開始してください。")
        finally:
            self.is_moving = False

    def on_key_3_contact_identify(self):
        """[3] キー: 3回接触同定シーケンスプレビュー開始 ➔ [3] または [Enter] で実機実行"""
        if self.is_moving:
            print("⚠️ ロボット移動中のため受け付けられません。")
            return

        if self.preview_action == "CONTACT_IDENTIFY":
            self.on_confirm_preview()
            return

        angles = [+30.0, +15.0, -10.0]
        traj = []
        home_q = CARRY_HOME_Q.copy()

        self.robot.fk(qs=home_q)
        tcp = self.gripper.tcp('grasp_center')
        home_pos = tcp.tf[:3, 3]
        home_rot = tcp.tf[:3, :3]

        cur_anim_q = home_q.copy()

        Y_BOTTOM_TCP = 0.115  # [m] 把持中心からアルミフレーム先端底面までの距離 (nrs_vr2と完全同一)

        for idx, ang in enumerate(angles):
            is_last = (idx == len(angles) - 1)
            tilt_rad = np.radians(ang)
            # ★ nrs_vr2 と完全同一: 世界座標系 X 軸正方向回り ([1, 0, 0]) ピッチ傾け
            R_rot_world = oum.rotmat_from_axangle(np.array([1.0, 0.0, 0.0]), tilt_rad)
            R_tilted = R_rot_world @ home_rot

            # 安全待機高さ (机面上 11.5cm 待機)
            z_c = +PROFILE_HEIGHT / 2.0 if ang > 0 else -PROFILE_HEIGHT / 2.0
            c_tcp = np.array([0.0, Y_BOTTOM_TCP, z_c])
            c_w_rel = R_tilted @ c_tcp
            standby_tcp_z = TABLE_Z + 0.115 - c_w_rel[2]
            p_standby = np.array([home_pos[0], home_pos[1], standby_tcp_z])

            sols_tilt = self.robot.ik(p_standby, R_tilted, tcp=tcp)
            q_tilt = pick_closest_ik(sols_tilt, cur_anim_q)
            if q_tilt is None:
                q_tilt = cur_anim_q

            # 机面接触位置 (最下点が机面 TABLE_Z に接触)
            contact_tcp_z = TABLE_Z - c_w_rel[2]
            p_contact = np.array([home_pos[0], home_pos[1], contact_tcp_z])
            sols_contact = self.robot.ik(p_contact, R_tilted, tcp=tcp)
            q_contact = pick_closest_ik(sols_contact, q_tilt)
            if q_contact is None:
                q_contact = q_tilt

            # 回避リフト位置 (机面から +8cm 垂直上昇)
            p_lift = p_contact.copy()
            p_lift[2] += 0.08
            sols_lift = self.robot.ik(p_lift, R_tilted, tcp=tcp)
            q_lift = pick_closest_ik(sols_lift, q_contact)
            if q_lift is None:
                q_lift = q_contact

            # 1. 前の姿勢から直接傾斜待機姿勢へ (12 frames)
            for i in range(12):
                s = (1.0 - np.cos(np.pi * (i / 11))) / 2.0
                traj.append((cur_anim_q + s * (q_tilt - cur_anim_q), PROFILE_WIDTH, True))

            # 2. 待機 -> 接地 (12 frames)
            for i in range(12):
                s = (1.0 - np.cos(np.pi * (i / 11))) / 2.0
                traj.append((q_tilt + s * (q_contact - q_tilt), PROFILE_WIDTH, True))

            # 3. contact 静止 (4 frames)
            for _ in range(4):
                traj.append((q_contact.copy(), PROFILE_WIDTH, True))

            # 4. 接地 -> +8cm 垂直上昇 (10 frames)
            for i in range(10):
                s = (1.0 - np.cos(np.pi * (i / 9))) / 2.0
                traj.append((q_contact + s * (q_lift - q_contact), PROFILE_WIDTH, True))

            if is_last:
                # 最後のステップのみホーム復帰 (12 frames)
                for i in range(12):
                    s = (1.0 - np.cos(np.pi * (i / 11))) / 2.0
                    traj.append((q_lift + s * (home_q - q_lift), PROFILE_WIDTH, True))
                cur_anim_q = home_q.copy()
            else:
                # 次のステップは上昇位置 q_lift から直接次角度へ回転！ (ホームに戻らない)
                cur_anim_q = q_lift.copy()

        self.preview_action = "CONTACT_IDENTIFY"
        self.preview_traj = traj
        self.preview_frame_idx = 0

        self.set_status("● 状態: 3回接触プレビュー中 - [3]または[Enter]で実機実行", color=(255, 215, 0, 255))
        self.set_banner("★【3回接触同定プレビュー中】 [3] または [Enter] で実機へ送信実行！ | [ESC] でキャンセル",
                        border_color=(230, 180, 40), text_color=(255, 230, 80, 255), bg_color=(36, 32, 20))

        print("\n" + "=" * 70)
        print("✓ 3回接触同定シーケンスのシミュレーションプレビューを開始しました！")
        print("  ・傾斜角度 : +30.0° ➔ +15.0° ➔ -10.0° の順に接地・退避")
        print("  ・シミュレーション画面で手先の傾きと机面タッチの様子をご確認ください。")
        print("  ・[3] または [Enter] / [Y] : 実機へ送信して3回接触同定を開始！")
        print("  ・[ESC] : キャンセル")
        print("=" * 70 + "\n")

    def _worker_execute_contact_and_identify(self):
        self.is_moving = True
        self.is_identifying = True
        print("\n========================================================")
        print(">>> [3] 3回接触同定シーケンスを開始します (nrs_vr2準拠)...")
        print("========================================================")

        angles = [+30.0, +15.0, -10.0]
        recorded_points = []

        total = len(angles)
        for i, ang in enumerate(angles):
            step = i + 1
            is_last = (step == total)
            print(f"\n--- ステップ {step}/{total}: 傾斜角度 {ang:+.1f}° (世界X軸回りピッチ) ---")
            res_contact = self.run_single_angle_contact(
                ang,
                step_num=step,
                total_steps=total,
                return_home=is_last,
                lift_after_contact_m=0.08
            )
            if res_contact is None:
                print(f"❌ ステップ {step} で接触検知に失敗または中断されました。")
                self.is_moving = False
                self.is_identifying = False
                return
            if isinstance(res_contact, tuple):
                contact_pt, tcp_pos, R_tilted = res_contact
            else:
                contact_pt, tcp_pos, R_tilted = res_contact, res_contact, np.eye(3)

            recorded_points.append({
                'step': step,
                'angle_deg': ang,
                'contact_world': contact_pt.tolist(),
                'tcp_world_pos': tcp_pos,
                'tcp_world_rot': R_tilted
            })

        self.set_status("● 状態: 誤差同定完了！ - [4]で精密設置実行", color=(63, 185, 80, 255))
        self.set_banner("✓ 誤差同定完了！ [4] キーで精密設置を実行してください",
                        border_color=(46, 160, 67), text_color=(86, 211, 100, 255), bg_color=(24, 32, 24))
        print("\n★ 全3回の接触が正常に完了しました！同定計算を行います...")
        id_res = self.compute_3x3_identification(recorded_points)
        self.x_identified = id_res

        self.plan_place()

        self.is_identifying = False
        self.is_moving = False

    def run_single_angle_contact(self, angle_deg, step_num, total_steps=3, return_home=False, lift_after_contact_m=0.08):
        tilt_rad = np.radians(angle_deg)

        self.robot.fk(qs=CARRY_HOME_Q)
        tcp = self.gripper.tcp('grasp_center')
        home_pos = tcp.tf[:3, 3]
        home_rot = tcp.tf[:3, :3]

        # ★ nrs_vr2 と完全同一: 世界座標系 X 軸正方向回り ([1, 0, 0]) ピッチ傾け
        R_rot_world = oum.rotmat_from_axangle(np.array([1.0, 0.0, 0.0]), tilt_rad)
        R_target_world = R_rot_world @ home_rot

        Y_BOTTOM_TCP = 0.115  # [m] 把持中心からアルミフレーム先端底面までの距離 (nrs_vr2準拠)
        z_c = +PROFILE_HEIGHT / 2.0 if angle_deg > 0 else -PROFILE_HEIGHT / 2.0
        c_tcp = np.array([0.0, Y_BOTTOM_TCP, z_c])
        c_w_rel = R_target_world @ c_tcp
        standby_tcp_z = TABLE_Z + 0.115 - c_w_rel[2]
        p_standby = np.array([home_pos[0], home_pos[1], standby_tcp_z])

        # 目標待機姿勢のフランジ基準 pose (ベース座標系)
        align_pose = self.get_flange_pose_from_tcp(p_standby, R_target_world)

        if self.rtde_c is not None:
            # 1. 前ステップの上昇位置から直接、今回の傾斜待機姿勢へ姿勢回転・移動 (ホームに戻らない！)
            print(f"  ・姿勢傾斜中 ({angle_deg:+.1f}°: 世界X軸回りピッチ)...")
            self.rtde_c.moveL(align_pose, 0.030, 0.060, False)
            time.sleep(0.4)

            # 2. 基準力覚バイアス測定 (静止状態・30サンプル平均: nrs_vr2準拠)
            print(f"  ・基準力覚バイアス測定中 ({angle_deg:+.1f}° 静止状態・重力補償適用)...")
            time.sleep(0.3)
            init_forces = []
            for _ in range(30):
                raw = np.array(self.rtde_r.getActualTCPForce()[:3])
                pose = self.rtde_r.getActualTCPPose()
                R_tcp_b = Rot.from_rotvec(pose[3:6]).as_matrix()
                net = self.compute_gravity_comp_force(R_tcp_b, raw)
                init_forces.append(net)
                time.sleep(0.008)
            F_start_net = np.mean(init_forces, axis=0)

            # 3. 鉛直直線下降目標 pose (moveL: 世界Z軸真下に最大 14cm 鉛直下降)
            current_flange_base = list(self.rtde_r.getActualTCPPose())
            descent_stroke_m = 0.140  # 14cm の確実な鉛直下降ストローク
            delta_world = np.array([0.0, 0.0, -descent_stroke_m])
            delta_base = self.R_base_world.T @ delta_world
            target_flange_pos_base = np.array(current_flange_base[:3]) + delta_base
            target_rotvec_base = current_flange_base[3:6]
            target_flange_down = list(np.concatenate([target_flange_pos_base, target_rotvec_base]))

            print(f">> 【鉛直直線下降中】 机面に向かって直線下降開始 (速度 1.2 cm/s, 力覚監視)...")
            contact_detected = False
            contact_force_val = 0.0
            stop_reason = ""
            start_time = time.time()
            consecutive_hits = 0

            filter_window = []
            window_size = 4
            history = []

            # ★ moveL で世界Z軸真下にまっすぐ鉛直直線下降！ (速度 0.012 m/s = 1.2 cm/s)
            self.rtde_c.moveL(target_flange_down, 0.012, 0.040, asynchronous=True)

            while time.time() - start_time < 22.0:
                raw_force = np.array(self.rtde_r.getActualTCPForce()[:3])
                curr_pose = self.rtde_r.getActualTCPPose()
                R_curr_b = Rot.from_rotvec(curr_pose[3:6]).as_matrix()
                net_force = self.compute_gravity_comp_force(R_curr_b, raw_force)

                filter_window.append(net_force)
                if len(filter_window) > window_size:
                    filter_window.pop(0)

                filtered_net = np.mean(filter_window, axis=0)
                df_base = filtered_net - F_start_net
                # ★ 世界座標系 Z 軸真上方向 (鉛直上向き) の正味接触反力！ (nrs_vr2準拠)
                df_world_z = float((self.R_base_world @ df_base)[2])

                now_t = time.time()
                elapsed = now_t - start_time
                f_smooth = df_world_z

                history.append((now_t, f_smooth))
                while history and (now_t - history[0][0] > 0.25):
                    history.pop(0)

                past_samples = [h[1] for h in history if 0.12 <= (now_t - h[0]) <= 0.18]
                f_baseline = past_samples[0] if past_samples else history[0][1]
                jump_fz = f_smooth - f_baseline

                # 接触判定 (初動 0.4s は加速度ノイズ対策でマスク: nrs_vr2準拠)
                if elapsed > 0.4:
                    hit = False
                    hit_reason_candidate = ""

                    # ★ 机面接触判定: 0.15s 間の垂直上向き反力ジャンプ +8.5N かつ 反力 >= 12.0N
                    if jump_fz >= 8.5 and f_smooth >= 12.0:
                        hit = True
                        hit_reason_candidate = f"★机面接触ジャンプ検知 (+{jump_fz:.1f}N/0.15s, 反力={f_smooth:.1f}N)"
                    elif f_smooth >= self.force_threshold_N:
                        hit = True
                        hit_reason_candidate = f"机面接触即時リミット到達 (反力={f_smooth:.1f}N >= {self.force_threshold_N:.1f}N)"

                    if hit:
                        consecutive_hits += 1
                        if consecutive_hits >= 2:
                            self.rtde_c.stopL(15.0)
                            contact_detected = True
                            contact_force_val = f_smooth
                            stop_reason = hit_reason_candidate
                            print(f"\n  ★★★【接触検知停止: ステップ {step_num} (角度 {angle_deg:+.1f}°)】★★★")
                            print(f"  ・判定理由     : {stop_reason}")
                            print(f"  ・検知接触反力 : +{contact_force_val:.2f} N (所要時間: {elapsed:.2f}s)")
                            time.sleep(0.2)
                            break
                    else:
                        consecutive_hits = 0

                time.sleep(0.005)

            if not contact_detected:
                self.rtde_c.stopL(15.0)
                print(f"\n⚠️ ステップ {step_num} で接触が検知されませんでした。")
                return None

            # 4. 接触直後の押し込み応力ゼロ化 (垂直微小リトラクト: nrs_vr2完全一致)
            retract_count = 0
            for _ in range(15):
                cur_raw = np.array(self.rtde_r.getActualTCPForce()[:3])
                cur_pose = self.rtde_r.getActualTCPPose()
                R_curr_b = Rot.from_rotvec(cur_pose[3:6]).as_matrix()
                cur_net = self.compute_gravity_comp_force(R_curr_b, cur_raw)
                cur_df = cur_net - F_start_net
                cur_world_fz = float((self.R_base_world @ cur_df)[2])
                if cur_world_fz <= 2.5:
                    break
                up_delta_world = np.array([0.0, 0.0, 0.0003])
                up_delta_base = self.R_base_world.T @ up_delta_world
                up_target = list(cur_pose)
                up_target[:3] = np.array(cur_pose[:3]) + up_delta_base
                self.rtde_c.moveL(up_target, 0.005, 0.05, False)
                retract_count += 1
                time.sleep(0.015)

            if retract_count > 0:
                print(f"  ✓ 押し込み応力を解放しました (垂直引き戻し量: {retract_count * 0.3:.2f} mm)")

            contact_q = np.array(self.rtde_r.getActualQ())
            self.robot.fk(qs=contact_q)
            c_pos = self.gripper.tcp('grasp_center').tf[:3, 3]
            contact_pt = np.array([c_pos[0], c_pos[1], TABLE_Z])

            # 接地点赤色球マーカー描画
            sp = ossop.sphere(pos=contact_pt, radius=0.006, rgb=ouc.BasicColor.RED)
            sp.attach_to(self.base.scene)
            self.contact_spheres.append(sp)

            # 5. 上昇動作: 最後のステップならホーム復帰、途中ステップなら 8cm 垂直上昇して終了 (ホームには戻らない！)
            if return_home:
                print(">> 3回接触完了！ ホームポジションへ復帰中...")
                cur_p = list(self.rtde_r.getActualTCPPose())
                up_w = np.array([0.0, 0.0, lift_after_contact_m])
                up_b = self.R_base_world.T @ up_w
                up_tgt = list(cur_p)
                up_tgt[:3] = np.array(cur_p[:3]) + up_b
                self.rtde_c.moveL(up_tgt, 0.030, 0.08, False)
                time.sleep(0.2)
                self.rtde_c.moveJ(CARRY_HOME_Q, SPEED_NORMAL, ACCEL_NORMAL)
                print("  ✓ ホームポジション到着完了。")
            else:
                print(f">> 次のステップのため机面から {lift_after_contact_m*100:.1f} cm 垂直上昇中 (ホームには戻らず直接次角度へ)...")
                cur_p = list(self.rtde_r.getActualTCPPose())
                up_w = np.array([0.0, 0.0, lift_after_contact_m])
                up_b = self.R_base_world.T @ up_w
                up_tgt = list(cur_p)
                up_tgt[:3] = np.array(cur_p[:3]) + up_b
                self.rtde_c.moveL(up_tgt, 0.030, 0.08, False)
                time.sleep(0.3)
                print(f"  ✓ {lift_after_contact_m*100:.1f} cm 垂直上昇完了。そのまま次の傾斜へ移行します。")

            return contact_pt, c_pos, R_target_world

        else:
            contact_tcp_z = TABLE_Z - c_w_rel[2]
            c_pos = np.array([home_pos[0], home_pos[1], contact_tcp_z])
            contact_pt = np.array([home_pos[0], home_pos[1], TABLE_Z])
            sp = ossop.sphere(pos=contact_pt, radius=0.006, rgb=ouc.BasicColor.RED)
            sp.attach_to(self.base.scene)
            self.contact_spheres.append(sp)
            return contact_pt, c_pos, R_target_world

    def compute_3x3_identification(self, recorded_points):
        """長浜理論 (3x3縮約版) によるアルミフレーム位置姿勢誤差同定計算"""
        if len(recorded_points) < 3:
            return None

        A_rows = []
        b_rows = []
        n_w = np.array([0.0, 0.0, 1.0])
        Y_BOTTOM_TCP = +0.115  # nrs_vr2準拠

        for res in recorded_points:
            ang = res["angle_deg"]
            tcp_w = res.get("tcp_world_pos", np.array(res["contact_world"]))
            R_w = res.get("tcp_world_rot", np.eye(3))

            z_contact = +PROFILE_HEIGHT / 2.0 if ang > 0 else -PROFILE_HEIGHT / 2.0
            c_tcp = np.array([0.0, Y_BOTTOM_TCP, z_contact])

            g_n_tcp = R_w.T @ n_w
            q_i = TABLE_Z - tcp_w[2]

            cross_cn = np.cross(c_tcp, g_n_tcp)
            row_3 = [g_n_tcp[1], g_n_tcp[2], cross_cn[0]]
            n_dot_c = float(np.dot(g_n_tcp, c_tcp))
            rhs_i = q_i - n_dot_c

            A_rows.append(row_3)
            b_rows.append(rhs_i)

        A = np.array(A_rows)
        b = np.array(b_rows)
        rank_A = np.linalg.matrix_rank(A)

        py_tcp_mm = 0.0
        pz_tcp_mm = 0.0
        wx_tcp_deg = 0.0

        if rank_A == 3:
            x_est = np.linalg.solve(A, b)
            py_tcp_mm = float(x_est[0] * 1000.0)
            pz_tcp_mm = float(x_est[1] * 1000.0)
            wx_tcp_deg = float(np.degrees(x_est[2]))

        delta_p_tcp = np.array([0.0, py_tcp_mm / 1000.0, pz_tcp_mm / 1000.0])
        R_err_tcp = oum.rotmat_from_euler(np.radians(wx_tcp_deg), 0.0, 0.0)

        res = {
            'delta_p': delta_p_tcp,
            'R_err': R_err_tcp,
            'py_tcp_mm': py_tcp_mm,
            'pz_tcp_mm': pz_tcp_mm,
            'wx_tcp_deg': wx_tcp_deg,
            'py_mm': py_tcp_mm,
            'pz_mm': pz_tcp_mm,
            'wx_deg': wx_tcp_deg,
            'recorded_points': recorded_points
        }
        print("\n【長浜理論 (3x3) 同定結果】")
        print(f"  ・把持並進誤差: py = {py_tcp_mm:+6.2f} mm, pz = {pz_tcp_mm:+6.2f} mm")
        print(f"  ・把持回転誤差: wx = {wx_tcp_deg:+6.2f} deg")
        print("========================================================")
        return res

    def plan_place(self):
        """推定誤差を相殺した退避エリア (X=0.45, Y=0.40) 精密設置動作計画 (nrs_vr2準拠)"""
        print("\n>>> 推定誤差を相殺した退避エリア設置動作計画を生成中 (nrs_vr2準拠)...")
        PLACE_XY = np.array([0.4500, 0.4000])

        delta_p_tcp = self.x_identified.get("delta_p", np.zeros(3))
        R_err_tcp = self.x_identified.get("R_err", np.eye(3))

        # 目標姿勢 (nrs_vr2準拠): 指非接触 6x16面が下面、長軸前後水平 (世界Y軸方向)
        x_obj_target = np.array([-1.0, 0.0, 0.0])
        y_obj_target = np.array([0.0, 1.0, 0.0])
        z_obj_target = np.array([0.0, 0.0, -1.0])
        R_obj_target = np.column_stack([x_obj_target, y_obj_target, z_obj_target])
        R_w_place = R_obj_target @ R_err_tcp.T

        # 把持中心から物体幾何中心への公称オフセット (nrs_vr2準拠: +0.035m)
        p_center_in_tcp_nom = np.array([0.0, 0.035, 0.0])

        # 1. 設置位置 (机面上 0.3cm = 3.0mm 空中)
        PLACE_AIR_HEIGHT_M = 0.003
        p_center_placed = np.array([
            PLACE_XY[0],
            PLACE_XY[1],
            TABLE_Z + PLACE_AIR_HEIGHT_M + (PROFILE_HEIGHT / 2.0)
        ])
        t_w_place = p_center_placed - R_w_place @ (delta_p_tcp + R_err_tcp @ p_center_in_tcp_nom)

        # 自然なシード角 (nrs_vr2準拠)
        seed_natural = np.radians([23.86, -195.12, 68.14, 194.97, 229.71, -302.02])

        tcp = self.gripper.tcp('grasp_center')
        sols_place = self.robot.ik(t_w_place, R_w_place, tcp=tcp)
        q_place = pick_closest_ik(sols_place, seed_natural)

        if q_place is None:
            print("❌ 設置姿勢の IK が解けませんでした。")
            return False

        # 2. 直上アプローチ位置 (退避エリア直上空 机面上 16cm)
        p_center_app = np.array([
            PLACE_XY[0],
            PLACE_XY[1],
            TABLE_Z + 0.160 + (PROFILE_HEIGHT / 2.0)
        ])
        t_w_app = p_center_app - R_w_place @ (delta_p_tcp + R_err_tcp @ p_center_in_tcp_nom)
        sols_app = self.robot.ik(t_w_app, R_w_place, tcp=tcp)
        q_app = pick_closest_ik(sols_app, q_place)

        # 3. リフト位置 (3回接触ホームから +12cm 垂直上昇)
        self.robot.fk(qs=CARRY_HOME_Q)
        t_w_home = tcp.tf[:3, 3]
        R_w_home = tcp.tf[:3, :3]
        t_w_lift = t_w_home + np.array([0.0, 0.0, 0.120])
        sols_lift = self.robot.ik(t_w_lift, R_w_home, tcp=tcp)
        q_lift = pick_closest_ik(sols_lift, CARRY_HOME_Q)
        if q_lift is None:
            q_lift = CARRY_HOME_Q.copy()

        self.place_plan_traj = {
            'lift_q': q_lift,
            'standby_q': q_app,
            'place_q': q_place
        }

        print("✓ 設置動作計画が完了しました！")
        print(f"  ・設置場所 : [{PLACE_XY[0]:.2f}, {PLACE_XY[1]:.2f}] (机面上 0.3cm)")
        print("  ・[4] キーを押すと精密設置プレビューを開始します。")
        return True

    def on_key_4_execute_place(self):
        """[4] キー: 精密設置動作プレビュー開始 ➔ [4] または [Enter] で実機実行"""
        if self.is_moving:
            print("⚠️ ロボット移動中のため受け付けられません。")
            return
        if self.place_plan_traj is None:
            print("⚠️ 設置計画が完了していません。[3] キーの同定を完了してください。")
            return

        if self.preview_action == "EXECUTE_PLACE":
            self.on_confirm_preview()
            return

        home_q = CARRY_HOME_Q.copy()
        q_lift = self.place_plan_traj['lift_q']
        q_app = self.place_plan_traj['standby_q']
        q_place = self.place_plan_traj['place_q']

        traj = []
        # Phase 1: home -> lift (+12cm垂直上昇, 15 frames)
        for i in range(15):
            s = (1.0 - np.cos(np.pi * (i / 14))) / 2.0
            traj.append((home_q + s * (q_lift - home_q), PROFILE_WIDTH, True))

        # Phase 2: lift -> standby/approach (退避エリア直上空へ水平搬送, 25 frames)
        for i in range(25):
            s = (1.0 - np.cos(np.pi * (i / 24))) / 2.0
            traj.append((q_lift + s * (q_app - q_lift), PROFILE_WIDTH, True))

        # Phase 3: standby -> place (机面上 0.3cm へ精密下降, 20 frames)
        for i in range(20):
            s = (1.0 - np.cos(np.pi * (i / 19))) / 2.0
            traj.append((q_app + s * (q_place - q_app), PROFILE_WIDTH, True))

        # Phase 4: place 静止 (5 frames)
        for _ in range(5):
            traj.append((q_place.copy(), PROFILE_WIDTH, True))

        # Phase 5: グリッパ全開 (8 frames)
        for i in range(8):
            w = PROFILE_WIDTH + (0.085 - PROFILE_WIDTH) * (i / 7)
            traj.append((q_place.copy(), w, False))

        # Phase 6: 上昇退避 place -> standby (15 frames)
        for i in range(15):
            s = (1.0 - np.cos(np.pi * (i / 14))) / 2.0
            traj.append((q_place + s * (q_app - q_place), 0.085, False))

        # Phase 7: standby -> VIEW_POSE_Q (20 frames)
        for i in range(20):
            s = (1.0 - np.cos(np.pi * (i / 19))) / 2.0
            traj.append((q_app + s * (VIEW_POSE_Q - q_app), 0.085, False))

        self.preview_action = "EXECUTE_PLACE"
        self.preview_traj = traj
        self.preview_frame_idx = 0

        self.set_status("● 状態: 精密設置プレビュー中 - [4]または[Enter]で実機実行", color=(255, 215, 0, 255))
        self.set_banner("★【精密設置プレビュー中】 [4] または [Enter] で実機へ送信実行！ | [ESC] でキャンセル",
                        border_color=(230, 180, 40), text_color=(255, 230, 80, 255), bg_color=(36, 32, 20))

        print("\n" + "=" * 70)
        print("✓ 精密設置動作のシミュレーションプレビューを開始しました！")
        print("  ・設置位置 : 退避エリア [X=0.45, Y=0.40] (机面上 0.3cm)")
        print("  ・姿勢     : 指非接触面が下面、長軸前後水平 (nrs_vr2準拠)")
        print("  ・動作内容 : 垂直リフト ➔ 水平搬送 ➔ 精密下降 ➔ グリッパ開放 ➔ 上昇退避 ➔ 観察姿勢")
        print("  ・[4] または [Enter] / [Y] : 実機へ送信して精密設置を実行！")
        print("  ・[ESC] : キャンセル")
        print("=" * 70 + "\n")

    def _worker_execute_place(self):
        self.is_moving = True
        print("\n>>> [4] 精密設置動作を開始します (nrs_vr2準拠)...")

        try:
            if self.rtde_c is not None:
                print("  ・Phase 1: ホームから +12cm 垂直上昇中...")
                self.rtde_c.moveJ(self.place_plan_traj['lift_q'], SPEED_SLOW, ACCEL_SLOW)
                time.sleep(0.2)

                print("  ・Phase 2: 退避エリア直上空 (机面上 16cm) へ上空水平搬送中...")
                self.rtde_c.moveJ(self.place_plan_traj['standby_q'], SPEED_NORMAL, ACCEL_NORMAL)
                time.sleep(0.2)

                print("  ・Phase 3: 机面上 0.3cm 空中へ精密下降中...")
                self.rtde_c.moveJ(self.place_plan_traj['place_q'], SPEED_SLOW, ACCEL_SLOW)
                time.sleep(0.5)

                print("  ・Phase 4: グリッパー全開 (85 mm) に開放中...")
                self.set_gripper_opening_safe(0.085)
                self.grip_ctrl.open()
                time.sleep(0.8)

                print("  ・Phase 5: 直上空へ上昇退避中...")
                self.rtde_c.moveJ(self.place_plan_traj['standby_q'], SPEED_SLOW, ACCEL_SLOW)
                time.sleep(0.2)

                print("  ・Phase 6: カメラ観察姿勢へ安全復帰中...")
                self.rtde_c.moveJ(VIEW_POSE_Q, SPEED_NORMAL, ACCEL_NORMAL)
            else:
                self.robot.fk(qs=self.place_plan_traj['place_q'])

            self.is_picked = False
            self.is_gripper_closed = False
            self.set_status("● 状態: 精密設置完了！", color=(63, 185, 80, 255))
            self.set_banner("✓ 精密設置完了！ [O]で開放 / [V]でカメラ観察姿勢へ復帰",
                            border_color=(46, 160, 67), text_color=(86, 211, 100, 255), bg_color=(24, 32, 24))
            print("✓ [4] 精密設置および退避が完了しました！全シーケンス終了です。")
        finally:
            self.is_moving = False

    def on_return_home(self):
        if self.is_moving:
            return
        threading.Thread(target=self._worker_return_home, daemon=True).start()

    def _worker_return_home(self):
        self.is_moving = True
        try:
            print("\n>>> ホーム姿勢へ移動中...")
            if self.rtde_c is not None:
                self.rtde_c.moveJ(CARRY_HOME_Q, SPEED_NORMAL, ACCEL_NORMAL)
            self.robot.fk(qs=CARRY_HOME_Q)
            print("✓ ホーム姿勢に到着しました。")
        finally:
            self.is_moving = False

    def on_open_gripper(self):
        """[O] キー: 実機グリッパー全開 (85 mm) に開放"""
        if self.is_moving:
            print("⚠️ ロボット移動中のためグリッパー操作を受け付けられません。")
            return

        self.is_picked = False
        self.is_gripper_closed = False
        self.set_gripper_opening_safe(0.085)
        print("\n>>> [O] 実機グリッパー全開 (85 mm) に開放中...")
        def _worker_open():
            res = self.grip_ctrl.open()
            pos_int, cur_w = self.grip_ctrl.get_pos()
            print(f"✓ [O] 実機グリッパー全開完了: POS={pos_int}, 実測幅={cur_w*1000:.1f} mm (応答: {res})")
            self.set_status("● 状態: グリッパ開放 - [C]で把持", color=(220, 220, 220, 255))
            self.set_banner("グリッパを開放しました。[C] キーで把持できます",
                            border_color=(150, 150, 150), text_color=(200, 200, 200, 255), bg_color=(30, 30, 30))
        threading.Thread(target=_worker_open, daemon=True).start()

    def on_close_gripper(self):
        """[C] キー: 実機のグリッパーでアルミフレームを把持 (常に最大力 force=255)"""
        if self.is_moving:
            print("⚠️ ロボット移動中のためグリッパー操作を受け付けられません。")
            return

        self.is_picked = True
        self.is_gripper_closed = True
        self.set_gripper_opening_safe(PROFILE_WIDTH)
        print(f"\n>>> [C] 実機グリッパー把持中 (物体幅: {PROFILE_WIDTH*1000:.1f} mm, 把持力: 最大 {GRIPPER_MAX_FORCE})...")
        def _worker():
            res = self.grip_ctrl.grip_object(PROFILE_WIDTH, force=GRIPPER_MAX_FORCE)
            pos_int, cur_w = self.grip_ctrl.get_pos()
            print(f"✓ [C] 実機グリッパー把持完了: POS={pos_int}, 実測幅={cur_w*1000:.1f} mm (応答: {res})")
            self.set_status("● 状態: 把持完了 - [2]で接触ホーム姿勢へ搬送", color=(63, 185, 80, 255))
            self.set_banner("✓ 把持完了！ [2] キーを押して接触ホーム姿勢へ搬送してください",
                            border_color=(46, 160, 67), text_color=(86, 211, 100, 255), bg_color=(24, 32, 24))
            print("★ 把持が完了しました！ [2] キーを押してリフト＆ホーム搬送を開始してください。")
        threading.Thread(target=_worker, daemon=True).start()

    def on_abort(self):
        if self.preview_action is not None:
            print("\n※ シミュレーションプレビューをキャンセルしました。")
            self.preview_action = None
            self.preview_traj = []
            # シミュレーション姿勢を復帰
            if self.rtde_r is not None:
                try:
                    cur_q = np.array(self.rtde_r.getActualQ())
                    self.robot.fk(qs=cur_q)
                    self.current_sim_q = cur_q.copy()
                except Exception:
                    pass
            self.set_gripper_opening_safe(0.085)
            self.set_status("● 状態: 待機中", color=(220, 220, 220, 255))
            self.set_banner("プレビューをキャンセルしました",
                            border_color=(150, 150, 150), text_color=(200, 200, 200, 255), bg_color=(30, 30, 30))
            return

        print("\n⚠️ 非常停止 / 中断が要求されました。")
        if self.rtde_c is not None:
            try:
                self.rtde_c.stopL(4.0)
            except Exception:
                pass
        self.is_moving = False
        self.is_identifying = False


# ===============================================================================
# メインエントリーポイント
# ===============================================================================
def main():
    print("=" * 70)
    print(" [one 版: UR3Dual 左腕 全自動ピック・3回接触・同定・設置シーケンス]")
    print("=" * 70)
    print("【操作フロー (全工程シミュレーションプレビュー対応)】:")
    print("  [V]        : 撮影待機 (View Pose) プレビュー開始 (確認後 [Enter])")
    print("  [S]        : 検出姿勢確定 ＆ ピック動作計画 ➔ プレビュー開始")
    print("  [1]        : ピック動作プレビュー ➔ [1] または [Enter] で実機実行")
    print("  [2]        : 搬送動作プレビュー ➔ [2] または [Enter] で実機実行")
    print("  [3]        : 3回接触同定プレビュー ➔ [3] または [Enter] で実機実行")
    print("  [4]        : 精密設置動作プレビュー ➔ [4] または [Enter] で実機実行")
    print("  [Enter]/[Y]: シミュレーション確認完了 ➔ 実機ロボットへ送信して実行")
    print("  [H]        : ホーム姿勢移動")
    print("  [O] / [C]  : グリッパー手動開閉")
    print("  [ESC]      : プレビューキャンセル / 中断")
    print("-------------------------------------------------------")

    # カメラ視点: 机面と双腕ロボットを斜め上空から広く見渡す視点
    base = ovw.World(cam_pos=(1.8, 0.5, 2.0), cam_lookat_pos=(0.45, 0.1, 1.15))
    app = IntegratedPickContactPlace(base)

    base.schedule_interval(app.update, interval=1.0 / 30.0)
    base.run()


if __name__ == '__main__':
    main()
