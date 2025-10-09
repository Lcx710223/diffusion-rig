###LCX251009，显示加农炮脸、POSE脸、YAW值。

import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 初始化 MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
connections = mp_face_mesh.FACEMESH_TESSELATION

# 加载图像并提取网格
img = cv2.imread('/content/ME42/ME42.png')  # 替换为你的图像路径
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_mesh.process(img_rgb)
landmarks = results.multi_face_landmarks[0].landmark
points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

# 姿态参数:POSE张量为6维，第2维是YAW，即水平转角。
pose_cannon = np.zeros((6,))
pose_target = np.array([-0.04336101, 0.76221186,  0.02132003,  0.02302069,  0.01214224, -0.09170481])

# 姿态变换函数（旋转 + 平移）
def apply_pose(points, pose):
    pitch, yaw, roll = pose[:3]
    tx, ty, tz = pose[3:]

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.array([tx, ty, tz])
    return (points @ R.T) + T

# 应用姿态
points_cannon = apply_pose(points, pose_cannon)
points_target = apply_pose(points, pose_target)

# 镜像 Y 轴（相机视角）
points_cannon[:, 1] *= -1
points_target[:, 1] *= -1

# 可视化：XY 平面 + 网格线
plt.figure(figsize=(10, 10))

# 加农炮脸（灰色线框）
for conn in connections:
    i, j = conn
    plt.plot([points_cannon[i, 0], points_cannon[j, 0]],
             [points_cannon[i, 1], points_cannon[j, 1]],
             color='gray', linewidth=0.5, alpha=0.5)

# 目标POSE脸（蓝色线框）
for conn in connections:
    i, j = conn
    plt.plot([points_target[i, 0], points_target[j, 0]],
             [points_target[i, 1], points_target[j, 1]],
             color='blue', linewidth=0.6, alpha=0.8)

# 计算水平转角（Yaw）
yaw_rad = pose_target[1]
yaw_deg = np.degrees(yaw_rad)
# 标注在图上
plt.text(0.17, -0.3, f"Yaw: {yaw_deg:.2f}°", fontsize=14, color='blue',
         bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))


plt.title("CANON FACE ： POSE FACE ")
plt.xlabel("X")
plt.ylabel("-Y")
plt.axis('equal')
plt.grid(True)
plt.show()
