"""
simulate_fusion_demo.py
========================
Demo autonome (pas besoin de video ni de modele YOLO) qui montre le gain
de la fusion multi-capteurs par rapport a un seul capteur, sur une cible
qui MANOEUVRE (ligne droite puis virage -> plus realiste qu'un drone qui
ne fait que suivre une ligne droite).

Scenario :
- Camera   : 30 Hz, bruit faible (sigma=3px), latence quasi nulle
- Capteur RF/radar : 5 Hz, bruit plus fort (sigma=12px) mais qui ne perd
  jamais la cible (utile la nuit / mauvaise meteo, contrairement a la
  camera), et avec 150ms de latence de traitement -> les mesures
  arrivent "en retard" par rapport a leur timestamp reel : c'est le
  cas hors-sequence gere par FusionTrack.fuse().

A la fin :
- on compare le RMSE de la position estimee : camera seule / RF seule / fusion
- on projette une trajectoire future (1.5s) a partir du dernier point connu
  et on affiche le cone d'incertitude
"""

import numpy as np
import matplotlib.pyplot as plt
from sensor_fusion import Measurement, FusionTrack

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. Trajectoire "verite terrain" : ligne droite puis virage serre
# ---------------------------------------------------------------------
DT_TRUE = 0.01
T_TOTAL = 6.0
times_true = np.arange(0, T_TOTAL, DT_TRUE)

true_pos = []
x, y = 0.0, 0.0
vx, vy = 40.0, 5.0
for t in times_true:
    if 3.0 < t < 4.0:  # virage
        theta = np.deg2rad(90) * (t - 3.0)
        speed = np.hypot(vx, vy)
        vx = speed * np.cos(theta + np.arctan2(5.0, 40.0))
        vy = speed * np.sin(theta + np.arctan2(5.0, 40.0))
    x += vx * DT_TRUE
    y += vy * DT_TRUE
    true_pos.append((x, y))
true_pos = np.array(true_pos)


def true_position_at(t):
    idx = min(int(t / DT_TRUE), len(true_pos) - 1)
    return true_pos[idx]


# ---------------------------------------------------------------------
# 2. Generation des mesures de chaque capteur (avec bruit + latence)
# ---------------------------------------------------------------------
def make_sensor_stream(hz, sigma, latency_s, t_end, dropout_windows=()):
    """dropout_windows: liste de (t_start, t_end) ou le capteur ne produit
    RIEN (occlusion / contre-jour / nuage pour la camera, par exemple)."""
    period = 1.0 / hz
    R = np.eye(2) * (sigma ** 2)
    stream = []  # (arrival_time, Measurement)
    t = 0.0
    while t < t_end:
        in_dropout = any(a <= t <= b for a, b in dropout_windows)
        if not in_dropout:
            tx, ty = true_position_at(t)
            zx = tx + rng.normal(0, sigma)
            zy = ty + rng.normal(0, sigma)
            meas = Measurement(t=t, x=zx, y=zy, R=R, sensor_id="")
            arrival = t + latency_s
            stream.append((arrival, meas))
        t += period
    return stream

# la camera perd la cible pendant 1.8s, pile pendant le virage : c'est le
# pire moment possible (c'est souvent le cas en vrai : virage serre =
# angle qui sort du champ ou contre-jour) -> scenario volontairement
# difficile pour bien montrer l'interet de la fusion
CAMERA_DROPOUT = [(2.6, 4.4)]

camera_stream = make_sensor_stream(hz=30, sigma=3.0,  latency_s=0.01, t_end=T_TOTAL,
                                    dropout_windows=CAMERA_DROPOUT)
for _, m in camera_stream:
    m.sensor_id = "camera"

radar_stream = make_sensor_stream(hz=5, sigma=12.0, latency_s=0.15, t_end=T_TOTAL)
for _, m in radar_stream:
    m.sensor_id = "radar_rf"

# fusionne les deux flux et trie par ORDRE D'ARRIVEE (pas par timestamp de
# mesure !) -> c'est ce qui se passe reellement en systeme temps reel
all_events = sorted(camera_stream + radar_stream, key=lambda e: e[0])

# ---------------------------------------------------------------------
# 3. Trois pistes : camera seule / radar seul / fusion
# ---------------------------------------------------------------------
t0, x0, y0 = 0.0, *true_position_at(0.0)
track_fusion = FusionTrack(t0, x0, y0)
track_cam_only = FusionTrack(t0, x0, y0)
track_radar_only = FusionTrack(t0, x0, y0)

fusion_path, cam_path, radar_path, gt_path, t_axis = [], [], [], [], []

for arrival, meas in all_events:
    track_fusion.fuse(meas)

    # les pistes mono-capteur avancent en "dead-reckoning" (predict seul,
    # pas d'update) quand ce n'est pas leur mesure -> comparaison honnete :
    # un tracker mono-capteur reel continuerait a extrapoler, pas a rester
    # fige, pendant les trous de son propre capteur.
    if meas.sensor_id == "camera":
        track_cam_only.fuse(meas)
        track_radar_only.predict(meas.t)
    else:
        track_radar_only.fuse(meas)
        track_cam_only.predict(meas.t)

    fusion_path.append(track_fusion.get_position())
    cam_path.append(track_cam_only.get_position())
    radar_path.append(track_radar_only.get_position())
    gt_path.append(true_position_at(meas.t))
    t_axis.append(meas.t)

fusion_path = np.array(fusion_path)
cam_path = np.array(cam_path)
radar_path = np.array(radar_path)
gt_path = np.array(gt_path)


def rmse(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))

print("=== RMSE position (pixels) vs verite terrain ===")
print(f"Camera seule  : {rmse(cam_path, gt_path):7.2f}")
print(f"Radar/RF seul : {rmse(radar_path, gt_path):7.2f}")
print(f"Fusion        : {rmse(fusion_path, gt_path):7.2f}")

# ---------------------------------------------------------------------
# 4. Prediction de trajectoire a partir du dernier point fusionne
# ---------------------------------------------------------------------
pred = track_fusion.predict_trajectory(horizon_s=1.5, dt=0.05)
pred_t   = [p[0] for p in pred]
pred_x   = [p[1] for p in pred]
pred_y   = [p[2] for p in pred]
pred_sx  = [p[3] for p in pred]
pred_sy  = [p[4] for p in pred]

true_future = np.array([true_position_at(t) for t in pred_t])
pred_arr = np.array(list(zip(pred_x, pred_y)))
pred_err_end = np.hypot(pred_arr[-1,0]-true_future[-1,0], pred_arr[-1,1]-true_future[-1,1])
print(f"\nErreur de prediction a +{pred_t[-1]-pred_t[0]:.1f}s : {pred_err_end:.1f} px "
      f"(cone d'incertitude 1-sigma final : {pred_sx[-1]:.1f} x {pred_sy[-1]:.1f} px)")

# ---------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(gt_path[:,0], gt_path[:,1], 'k-', lw=2, label="Verite terrain")
ax.plot(cam_path[:,0], cam_path[:,1], color='tab:blue', alpha=0.6, label="Camera seule (30Hz, bruit faible)")
ax.plot(radar_path[:,0], radar_path[:,1], color='tab:orange', alpha=0.6, label="RF/radar seul (5Hz, bruit fort, retard 150ms)")
ax.plot(fusion_path[:,0], fusion_path[:,1], color='tab:green', lw=2, label="Fusion (camera + RF)")
ax.set_title("Fusion multi-capteurs vs mono-capteur")
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
ax.legend(loc="best", fontsize=8)
ax.axis("equal")

ax = axes[1]
ax.plot(gt_path[:,0], gt_path[:,1], 'k-', lw=1, alpha=0.3, label="Trajectoire passee (verite)")
ax.plot(fusion_path[:,0], fusion_path[:,1], color='tab:green', lw=2, label="Estimation fusionnee")
ax.plot(true_future[:,0], true_future[:,1], 'k--', lw=2, label="Futur reel")
ax.plot(pred_x, pred_y, color='tab:red', lw=2, label="Trajectoire predite")
for i in range(0, len(pred_x), 3):
    ax.add_patch(plt.matplotlib.patches.Ellipse(
        (pred_x[i], pred_y[i]), width=2*pred_sx[i], height=2*pred_sy[i],
        color='tab:red', alpha=0.12))
ax.set_title("Prediction de trajectoire + cone d'incertitude (1-sigma)")
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
ax.legend(loc="best", fontsize=8)
ax.axis("equal")

plt.tight_layout()
plt.savefig("fusion_demo.png", dpi=140)
print("\nGraphique sauvegarde -> fusion_demo.png")
