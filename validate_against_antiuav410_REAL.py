"""
validate_against_antiuav410_REAL.py
=====================================
Validation de FusionTrack sur une VRAIE sequence Anti-UAV410 (donnees
thermiques infrarouges reelles, pas de simulation).

Structure attendue (confirmee via le loader officiel datasets/antiuav410.py
du repo HwangBo94/Anti-UAV410) :

  AntiUAV410/test/<nom_sequence>/
      000001.jpg, 000002.jpg, ...   <- frames (pas utilisees ici, YOLO
                                        n'est pas entraine sur de l'IR)
      IR_label.json                  <- {"gt_rect": [[x,y,w,h], ...],
                                          "exist": [1,1,0,...]}

Les attributs de difficulte (Occlusion, Fast Motion, etc.) sont au niveau
de la SEQUENCE ENTIERE dans annos/test/att/<nom_sequence>.txt (10 flags
0/1 dans l'ordre : Thermal Crossover, Out-of-View, Scale Variation,
Fast Motion, Occlusion, Dynamic Background Clutter, Tiny Size, Small Size,
Medium Size, Normal Size). On les utilise pour choisir une sequence
interessante et pour etiqueter le rapport, mais la vraie difficulte
frame-par-frame vient de 'exist' (visible / pas visible) et des sauts de
vitesse observes dans gt_rect (proxy pour "mouvement rapide").

USAGE :
    python validate_against_antiuav410_REAL.py --seq /chemin/vers/AntiUAV410/test/03_3780_0001-1499
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sensor_fusion import Measurement, FusionTrack

ATTR_NAMES = ["Thermal_Crossover", "Out_of_View", "Scale_Variation",
              "Fast_Motion", "Occlusion", "Dynamic_Background_Clutter",
              "Tiny_Size", "Small_Size", "Medium_Size", "Normal_Size"]

rng = np.random.default_rng(0)


def load_sequence(seq_dir):
    """Charge gt_rect + exist depuis IR_label.json d'une sequence reelle."""
    label_path = os.path.join(seq_dir, "IR_label.json")
    with open(label_path, "r") as f:
        label = json.load(f)
    gt_rect = label["gt_rect"]           # liste de [x,y,w,h]
    exist = np.array(label.get("exist", [1] * len(gt_rect)), dtype=bool)
    # Certaines frames non-visibles stockent quand meme une bbox
    # "placeholder" du type [0,0,0,0] au lieu d'une liste vide -> on se
    # base sur le flag 'exist' officiel, pas sur le contenu de gt_rect,
    # pour decider si une position est valide.
    cx = np.full(len(gt_rect), np.nan)
    cy = np.full(len(gt_rect), np.nan)
    for i, r in enumerate(gt_rect):
        if exist[i] and r and (r[2] > 0 or r[3] > 0):
            cx[i] = r[0] + r[2] / 2
            cy[i] = r[1] + r[3] / 2
    return cx, cy, exist


def load_attributes(seq_name, att_dir):
    """Charge les 10 flags globaux de difficulte pour cette sequence,
    depuis annos/test/att/<seq_name>.txt (meme nom que le dossier)."""
    path = os.path.join(att_dir, seq_name + ".txt")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        vals = [int(v) for v in f.read().strip().split(",")]
    return dict(zip(ATTR_NAMES, vals))


def estimate_fast_motion_frames(cx, cy, exist, fps=30, speed_threshold_px_s=None):
    """L'attribut Fast_Motion n'est pas donne frame-par-frame -> on le
    derive nous-memes : frames ou la vitesse instantanee depasse un seuil
    (moyenne + 2 ecarts-types du mouvement de la sequence)."""
    n = len(cx)
    speed = np.zeros(n)
    for i in range(1, n):
        if exist[i] and exist[i - 1]:
            speed[i] = np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) * fps
    valid_speed = speed[speed > 0]
    if speed_threshold_px_s is None:
        speed_threshold_px_s = valid_speed.mean() + 2 * valid_speed.std() \
            if len(valid_speed) else 1e9
    return speed > speed_threshold_px_s, speed_threshold_px_s


def make_measurements(cx, cy, exist, fps, sigma_px=2.5, miss_rate=0.0):
    """Simule le flux de detections camera a partir de la verite-terrain
    reelle (bruit de mesure raisonnable type YOLO). exist=False -> pas de
    mesure (occlusion / hors champ REELLE, pas simulee)."""
    R = np.eye(2) * (sigma_px ** 2)
    measurements = []
    for i in range(len(cx)):
        if not exist[i] or np.isnan(cx[i]):
            continue
        if miss_rate > 0 and rng.random() < miss_rate:
            continue
        t = i / fps
        zx = cx[i] + rng.normal(0, sigma_px)
        zy = cy[i] + rng.normal(0, sigma_px)
        measurements.append((t, zx, zy, R, i))
    return measurements


def run_validation(seq_dir, att_dir=None, fps=30):
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    cx, cy, exist = load_sequence(seq_dir)
    n_frames = len(cx)
    print(f"Sequence: {seq_name}  ({n_frames} frames, "
          f"{exist.sum()} visibles / {(~exist).sum()} occlusion-ou-hors-champ)")

    attrs = load_attributes(seq_name, att_dir) if att_dir else {}
    if attrs:
        active = [k for k, v in attrs.items() if v == 1]
        print(f"Attributs actifs (dataset officiel) : {', '.join(active) if active else 'aucun'}")

    fast_mask, thresh = estimate_fast_motion_frames(cx, cy, exist, fps)
    print(f"Fast-motion detecte (seuil derive) : {fast_mask.sum()} frames "
          f"(seuil={thresh:.1f} px/s)\n")

    measurements = make_measurements(cx, cy, exist, fps)
    if not measurements:
        print("Aucune mesure exploitable dans cette sequence.")
        return

    t0, x0, y0, _, idx0 = measurements[0]
    track = FusionTrack(t0, x0, y0)

    est = {idx0: (x0, y0)}
    prev_idx = idx0
    for (t, zx, zy, R, idx) in measurements[1:]:
        for gap_idx in range(prev_idx + 1, idx):
            px, py = track.predict(gap_idx / fps)
            est[gap_idx] = (px, py)
        meas = Measurement(t=t, x=zx, y=zy, R=R, sensor_id="camera_ir")
        track.fuse(meas)
        px, py = track.get_position()
        est[idx] = (px, py)
        prev_idx = idx

    def err_at(i):
        if i not in est or not exist[i] or np.isnan(cx[i]):
            return None
        ex, ey = est[i]
        return float(np.hypot(ex - cx[i], ey - cy[i]))

    normal_err, fast_err = [], []
    for i in range(n_frames):
        e = err_at(i)
        if e is None:
            continue
        (fast_err if fast_mask[i] else normal_err).append(e)

    reacq_err = []
    for i in range(1, n_frames):
        if exist[i] and not exist[i - 1] and i in est:
            ex, ey = est[i]
            reacq_err.append(float(np.hypot(ex - cx[i], ey - cy[i])))

    def rmse(lst):
        return float(np.sqrt(np.mean(np.square(lst)))) if lst else float("nan")

    print("=== Resultats ===")
    print(f"{'Categorie':<25} {'n':>6} {'RMSE (px)':>11}")
    print("-" * 45)
    print(f"{'Normal':<25} {len(normal_err):>6} {rmse(normal_err):>11.2f}")
    print(f"{'Fast motion (derive)':<25} {len(fast_err):>6} {rmse(fast_err):>11.2f}")
    print(f"{'Ré-acquisition post-occl.':<25} {len(reacq_err):>6} {rmse(reacq_err):>11.2f}")
    print("-" * 45)
    all_err = normal_err + fast_err
    print(f"{'TOUT':<25} {len(all_err):>6} {rmse(all_err):>11.2f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    valid = exist & ~np.isnan(cx)
    ax.plot(cx[valid], cy[valid], 'k-', lw=1.2, alpha=0.4, label="Verite terrain")
    est_x = np.array([est[i][0] for i in sorted(est)])
    est_y = np.array([est[i][1] for i in sorted(est)])
    ax.plot(est_x, est_y, color='tab:green', lw=1.2, alpha=0.8, label="FusionTrack (estime)")
    occl_idx = np.where(~exist)[0]
    if len(occl_idx):
        ax.scatter(cx[np.clip(occl_idx - 1, 0, n_frames - 1)],
                   cy[np.clip(occl_idx - 1, 0, n_frames - 1)],
                   color='red', s=15, zorder=5, label="Debut occlusion/hors-champ")
    ax.set_title(f"Validation reelle — {seq_name}")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    out_path = f"validation_{seq_name}.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nGraphique sauvegarde -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True,
                        help="Chemin vers le dossier de la sequence, ex: "
                             "AntiUAV410/test/03_3780_0001-1499")
    parser.add_argument("--att_dir", default=None,
                        help="Chemin vers annos/test/att (optionnel)")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    run_validation(args.seq, args.att_dir, args.fps)