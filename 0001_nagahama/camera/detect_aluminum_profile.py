# -*- coding: utf-8 -*-
"""
===============================================================================
[one フレームワーク版: アルミフレーム リアルタイム物体検出＆3D初期姿勢推定]
-------------------------------------------------------------------------------
ファイル名: detect_aluminum_profile.py
対応フレームワーク: one (MuJoCo / pyglet)

【概要】:
  1. 机上に置かれたアルミフレーム (6cm x 6cm x 16cm, 上面 6cm x 16cm) を
     カメラ画像からマーカーなしで自動検出し、外接回転矩形と向きを特定します。
  2. カメラ内部・外部パラメータと「机面からの高さ (6cm)」による平面拘束を用い、
     アルミフレームの 3次元ワールド位置 (X, Y, Z [m]) および ヨー角 (theta [deg]) を推定します。
  3. 2頂点の3D逆投影によるパースペクティブ補正を行い、歪みのない真の長軸角度を算出します。
===============================================================================
"""

import os
import sys
import yaml
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import one.utils.math as oum
import one.robots.manipulators.universal_robots.ur3.ur3 as ormuu3
from one.robots.end_effectors.robotiq.rtq2f85.rtq2f85 import Rtq2F85

try:
    from rtde_receive import RTDEReceiveInterface
    HAS_UR_RTDE = True
except ImportError:
    HAS_UR_RTDE = False

DEVICE_ID = 12
ROBOT_IP = '10.2.0.50'
TABLE_Z = 1.0940           # 机面天板のワールド Z 座標 [m]
PROFILE_HEIGHT = 0.060     # アルミフレームの高さ [m] (6cm)
TARGET_PLANE_Z = TABLE_Z + PROFILE_HEIGHT  # 上面のワールド Z 座標 (1.1540 m)

VIEW_HOME_Q = np.deg2rad([-17.7, -209.6, 42.6, 273.9, 227.6, -245.8])


def build_ur3_dual_lft():
    """one フレームワークで UR3Dual 左腕 + Rtq2F85 モデルを構築"""
    base_pos = np.array([0.0, 0.258485281374, 1.61051471863], dtype=np.float32)
    base_rotmat = oum.rotmat_from_euler(-3.0 * np.pi / 4.0, 0.0, 0.0)
    robot = ormuu3.UR3(pos=base_pos, rotmat=base_rotmat)

    gripper = Rtq2F85()
    # ★ カップリング回転は 0.0 (実機の開閉軸と完全一致)
    loc_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, 0.049962], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(0.0, 0.0, 0.0),
    )
    robot.mount(gripper, robot.runtime_lnks[-1], loc_tf, update=True)
    return robot, gripper


class ProfileDetector:
    def __init__(self, cam_yaml=None, hand_eye_yaml=None):
        if cam_yaml is None:
            cam_yaml = os.path.join(current_dir, 'camera_params.yaml')
        if hand_eye_yaml is None:
            hand_eye_yaml = os.path.join(current_dir, 'hand_eye_params.yaml')

        with open(cam_yaml, 'r') as f:
            c_info = yaml.safe_load(f)
        self.K = np.array(c_info['camera_matrix'], dtype=np.float64)
        self.dist = np.array(c_info['distortion_coefficients'], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        with open(hand_eye_yaml, 'r') as f:
            h_info = yaml.safe_load(f)
        self.T_c2g = np.array(h_info['T_cam2gripper'], dtype=np.float64)

    def detect_in_frame(self, frame, cam_pos_w, cam_rot_w):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blur, 40, 120)

        # 外周除外マスク (机端やロボットアームの映り込みを除外)
        edges[:70, :] = 0
        edges[440:, :] = 0
        edges[:, :120] = 0
        edges[:, 580:] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_target = None
        max_score = 0

        for cnt in contours:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90
            aspect = rw / (rh + 1e-5)
            rect_area = rw * rh

            if not (80 < cx < 560 and 60 < cy < 440):
                continue

            if 2500 < rect_area < 25000 and 1.6 <= aspect <= 4.2:
                solidity = hull_area / (rect_area + 1e-5)
                if solidity > 0.50:
                    score = solidity / (1.0 + abs(aspect - 2.67))
                    if score > max_score:
                        max_score = score
                        best_target = (rect, aspect, angle, cnt, solidity)

        if best_target is None:
            return None, edges

        rect, aspect, angle, cnt, solidity = best_target
        (cx, cy), (rw, rh), _ = rect
        box = np.int64(cv2.boxPoints(rect))

        d01 = np.linalg.norm(box[0] - box[1])
        d12 = np.linalg.norm(box[1] - box[2])
        p_a, p_b = (box[0], box[1]) if d01 > d12 else (box[1], box[2])

        def project_pixel_to_plane(u, v):
            pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
            pt_u = cv2.undistortPoints(pt, self.K, self.dist, P=self.K)[0, 0]
            ray = self.K_inv @ np.array([pt_u[0], pt_u[1], 1.0])
            ray = ray / np.linalg.norm(ray)
            ray_w = cam_rot_w @ ray
            if abs(ray_w[2]) < 1e-4:
                return None
            s = (TARGET_PLANE_Z - cam_pos_w[2]) / ray_w[2]
            return cam_pos_w + s * ray_w

        pos_top_w = project_pixel_to_plane(cx, cy)
        pos_a_w = project_pixel_to_plane(p_a[0], p_a[1])
        pos_b_w = project_pixel_to_plane(p_b[0], p_b[1])

        if pos_top_w is None or pos_a_w is None or pos_b_w is None:
            return None, edges

        vec_long = pos_b_w - pos_a_w
        vec_long[2] = 0.0
        vec_len = np.linalg.norm(vec_long)
        if vec_len < 1e-4:
            return None, edges

        vec_long /= vec_len
        obj_yaw_deg = np.rad2deg(np.arctan2(vec_long[1], vec_long[0]))
        while obj_yaw_deg > 90.0:
            obj_yaw_deg -= 180.0
        while obj_yaw_deg <= -90.0:
            obj_yaw_deg += 180.0

        pos_center_w = np.copy(pos_top_w)
        pos_center_w[2] = TABLE_Z + PROFILE_HEIGHT / 2.0
        grasp_yaw_deg = (obj_yaw_deg + 90.0 + 180.0) % 360.0 - 180.0

        obj_info = {
            'pos_world': pos_top_w,
            'pos_center_world': pos_center_w,
            'yaw_deg': obj_yaw_deg,
            'grasp_yaw_deg': grasp_yaw_deg,
            'aspect': aspect,
            'rect': rect,
            'box_pts': box,
            'pixel_center': (cx, cy)
        }
        return obj_info, edges
