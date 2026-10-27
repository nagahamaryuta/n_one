# 0001_nagahama: one フレームワーク版 実機ロボット制御＆3回接触同定システム

旧 `nrs_vr2` (WRS / Panda3D) から新フレームワーク `n_one` (MuJoCo / pyglet) への移行プロジェクトです。

---

## 1. 動作環境 (推奨 Python 環境)
```bash
conda activate nagahama_env_py312
```

---

## 2. 実行方法

### ① カメラ物体検出・ピック・3回接触・精密設置 (フル自動統合シーケンス)
```bash
cd /home/wrs/PycharmProjects/n_one/0001_nagahama/three_contact_pose_id
/home/wrs/anaconda3/envs/nagahama_env_py312/bin/python integrated_pick_contact_place.py
```

### ② カメラでのアルミフレームリアルタイム検出・初期姿勢推定単体テスト
```bash
cd /home/wrs/PycharmProjects/n_one/0001_nagahama/camera
/home/wrs/anaconda3/envs/nagahama_env_py312/bin/python detect_aluminum_profile.py
```

---

## 3. 操作キー一覧

| キー | 動作内容 |
| :---: | :--- |
| **`[S]`** | カメラ検出位置・向きを確定保存 ＆ ピック動作の自動計画 |
| **`[1]`** | ピック動作の実行（アプローチ ➔ 下降 ➔ グリッパ把持） |
| **`[2]`** | 把持した物体の持ち上げ（+15cm） ➔ 3回接触ホーム姿勢への安全搬送 |
| **`[3]`** | **3回接触同定シーケンス実行**（各角度傾斜 ➔ 鉛直真下下降 ➔ 反力検知 ➔ 長浜理論3x3同定 ➔ 設置計画） |
| **`[4]`** | 精密設置動作の実行（同定誤差を相殺して水平接地 ➔ グリッパ開放 ➔ 退避） |
| **`[V]`** | 撮影待機姿勢 (View Pose) へ移動 |
| **`[H]`** | 待機ホーム姿勢へ移動 |
| **`[O]` / `[C]`** | グリッパー手動開閉 |
| **`[ESC]` / `[q]`** | 中断 / 非常停止 |

---

## 4. ディレクトリ構成
- `camera/`:
  - `detect_aluminum_profile.py`: カメラ検出・姿勢推定モジュール
  - `camera_params.yaml` / `.npz`: RealSense カメラ内部パラメータ
  - `hand_eye_params.yaml` / `.npz`: ハンドアイキャリブレーション行列
- `three_contact_pose_id/`:
  - `integrated_pick_contact_place.py`: one 対応・フル自動統合スクリプト
  - `tool_gravity_calib.json`: FT300 センサ重力補償パラメータ
  - `ft300_calib.json`: センサ原点オフセット
- `meshes/`: アルミフレーム等の 3D モデル
