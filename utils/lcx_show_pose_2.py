###LCX251009显示源图+加农脸+姿态脸+水平转角值。

import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
connections = mp_face_mesh.FACEMESH_TESSELATION

# 姿态参数:POSE为6维向量，第2维是YAW，即水平转角：
pose_cannon = np.zeros((6,))
pose_target = np.array([-0.04336101, 0.76221186,  0.02132003,  0.02302069,  0.01214224, -0.09170481])

# 加载原始图像
img_orig = cv2.imread('/content/ME42/ME420.png')
img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

# 加载网格图像
img_mesh = cv2.imread('/content/ME42/ME42.png')
img_mesh_rgb = cv2.cvtColor(img_mesh, cv2.COLOR_BGR2RGB)
results = face_mesh.process(img_mesh_rgb)
landmarks = results.multi_face_landmarks[0].landmark
points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

# 姿态变换函数
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

# 应用姿态并镜像 Y 轴
points_cannon = apply_pose(points, pose_cannon)
points_target = apply_pose(points, pose_target)
points_cannon[:, 1] *= -1
points_target[:, 1] *= -1

# 计算水平转角，并在图示上显示出来：
yaw_deg = np.degrees(pose_target[1])

# 创建并列图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 左图：原始图像
axs[0].imshow(img_orig_rgb)
axs[0].set_title("SOURCE FACE")
axs[0].axis('off')

# 中图：加农炮脸网格
axs[1].set_title("CANON FACE")
for conn in connections:
    i, j = conn
    axs[1].plot([points_cannon[i, 0], points_cannon[j, 0]],
                [points_cannon[i, 1], points_cannon[j, 1]],
                color='gray', linewidth=0.5, alpha=0.5)
axs[1].axis('equal')
axs[1].grid(True)

# 右图：目标POSE脸网格
axs[2].set_title("POSE INSTANCE FACE")
for conn in connections:
    i, j = conn
    axs[2].plot([points_target[i, 0], points_target[j, 0]],
                [points_target[i, 1], points_target[j, 1]],
                color='blue', linewidth=0.6, alpha=0.8)
axs[2].axis('equal')
axs[2].grid(True)

# 添加红色 Yaw 标注（标题下方）
fig.suptitle("CANON FACE : POSE FACE", fontsize=16)
fig.text(0.73, 0.8, f"Yaw: {yaw_deg:.2f}°", fontsize=14, color='red', ha='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

plt.tight_layout()
plt.show()
