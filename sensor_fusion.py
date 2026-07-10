"""
sensor_fusion.py
=================
Extension du tracker Kalman de Drone_Shaed_AI vers de la vraie fusion
multi-capteurs asynchrone, + prédiction de trajectoire.

Ce que ça ajoute par rapport au KalmanDrone existant :

1. Modele a acceleration constante (CA) au lieu de vitesse constante (CV)
   -> etat = [x, y, vx, vy, ax, ay]
   -> meilleure prediction quand la cible manoeuvre (un drone qui vire
      n'est pas bien modelise par une simple ligne droite).

2. Fusion multi-capteurs avec R (bruit) different par capteur
   -> une camera est precise mais bruitee sur la distance estimee,
      un capteur RF/radar est moins precis en position mais insensible
      a la lumiere / meteo. Fusionner les deux reduit l'incertitude
      globale (c'est exactement le probleme que Starlink/Starshield
      doivent resoudre pour du tracking d'assets).

3. Gestion des mesures hors-sequence (Out-Of-Sequence Measurements, OOSM)
   -> un capteur plus lent (ex: 150ms de latence) livre une mesure qui
      concerne un instant DEJA passe par rapport au dernier update.
      On ne peut pas juste l'appliquer maintenant sans "retro-corriger"
      l'estimation -> on garde un historique d'etats et on rejoue le
      filtre depuis ce point (technique standard en fusion de capteurs
      aerospatiale / defense).

4. Prediction de trajectoire avec cone d'incertitude
   -> on propage le modele CA dans le futur sans nouvelle mesure et on
      fait grandir l'ellipse de covariance -> utile pour anticiper une
      trajectoire (ex: estimer ou sera la cible dans 2s si le capteur
      la perd de vue).
"""

import numpy as np
from bisect import bisect_right


class Measurement:
    """Une mesure de position venant d'un capteur donne."""
    __slots__ = ("t", "z", "R", "sensor_id")

    def __init__(self, t, x, y, R, sensor_id):
        self.t = t                                   # timestamp (secondes)
        self.z = np.array([[x], [y]], dtype=float)    # position mesuree
        self.R = R                                    # covariance de bruit 2x2
        self.sensor_id = sensor_id


class FusionTrack:
    """
    Une piste unique, mise a jour par des mesures venant de N capteurs
    arrivant potentiellement dans le desordre (a cause des latences).

    Etat (6D) : [x, y, vx, vy, ax, ay]^T
    """

    # Bruit de process : incertitude sur le modele lui-meme (a quel point
    # on "croit" que l'acceleration reste constante). A augmenter si la
    # cible manoeuvre beaucoup, a diminuer si le mouvement est tres regulier.
    PROCESS_NOISE_STD = 4.0

    def __init__(self, t0, x0, y0, history_len=200):
        self.x = np.array([[x0], [y0], [0.], [0.], [0.], [0.]], dtype=float)
        self.P = np.eye(6, dtype=float) * 500.0
        self.t = t0

        # historique (t, x, P) pour pouvoir "rembobiner" en cas de mesure
        # hors-sequence -> c'est ce qui permet la fusion asynchrone propre
        self._history = [(t0, self.x.copy(), self.P.copy())]
        self._history_len = history_len

        # tampon des mesures deja appliquees, dans l'ordre de leur timestamp
        # (pas de leur arrivee !) pour pouvoir les "rejouer" apres un rewind
        self._applied = []

    # ------------------------------------------------------------------
    # Modele dynamique (acceleration constante)
    # ------------------------------------------------------------------
    @staticmethod
    def _F(dt):
        return np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

    @classmethod
    def _Q(cls, dt):
        q = cls.PROCESS_NOISE_STD ** 2
        # bruit de process discretise pour un modele a acceleration constante
        G = np.array([[dt**2/2, 0], [0, dt**2/2],
                      [dt, 0], [0, dt],
                      [1, 0], [0, 1]], dtype=float)
        return G @ (np.eye(2) * q) @ G.T

    _H = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0]], dtype=float)

    def _predict_from(self, x, P, t_from, t_to):
        dt = t_to - t_from
        if dt <= 0:
            return x, P
        F = self._F(dt)
        x = F @ x
        P = F @ P @ F.T + self._Q(dt)
        return x, P

    def _update(self, x, P, meas: Measurement):
        y = meas.z - self._H @ x
        S = self._H @ P @ self._H.T + meas.R
        K = P @ self._H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ self._H) @ P
        return x, P

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def predict(self, t_now):
        """Avance l'estimation courante jusqu'a t_now (sans nouvelle mesure)."""
        self.x, self.P = self._predict_from(self.x, self.P, self.t, t_now)
        self.t = t_now
        return self.get_position()

    def fuse(self, meas: Measurement):
        """
        Integre une mesure, MEME SI elle concerne un instant deja passe
        par rapport a self.t (mesure hors-sequence / capteur en retard).
        """
        if meas.t >= self.t:
            # cas simple : mesure "dans le present ou futur immediat"
            self.x, self.P = self._predict_from(self.x, self.P, self.t, meas.t)
            self.x, self.P = self._update(self.x, self.P, meas)
            self.t = meas.t
            self._checkpoint()
            self._applied.append(meas)
            return

        # --- mesure en retard : on rembobine et on rejoue --------------
        times = [h[0] for h in self._history]
        idx = bisect_right(times, meas.t) - 1
        idx = max(idx, 0)
        t0, x0, P0 = self._history[idx]

        # rejoue toutes les mesures deja connues qui sont APRES t0,
        # en y inserant la nouvelle mesure a la bonne place chronologique
        to_replay = sorted(
            [m for m in self._applied if m.t > t0] + [meas],
            key=lambda m: m.t
        )

        x, P, t = x0, P0, t0
        for m in to_replay:
            x, P = self._predict_from(x, P, t, m.t)
            x, P = self._update(x, P, m)
            t = m.t

        # on ramene l'estimation au present (self.t d'origine)
        x, P = self._predict_from(x, P, t, self.t)
        self.x, self.P = x, P
        self._applied.append(meas)
        self._applied.sort(key=lambda m: m.t)

    def _checkpoint(self):
        self._history.append((self.t, self.x.copy(), self.P.copy()))
        if len(self._history) > self._history_len:
            self._history.pop(0)
            # purge aussi les mesures trop vieilles pour rester bornes
            cutoff = self._history[0][0]
            self._applied = [m for m in self._applied if m.t >= cutoff]

    def get_position(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_velocity(self):
        return float(self.x[2, 0]), float(self.x[3, 0])

    def get_acceleration(self):
        return float(self.x[4, 0]), float(self.x[5, 0])

    # ------------------------------------------------------------------
    # Prediction de trajectoire (le "morceau de bravoure" pour l'entretien)
    # ------------------------------------------------------------------
    def predict_trajectory(self, horizon_s, dt=0.1):
        """
        Projette la trajectoire future sans nouvelle mesure.
        Retourne une liste de (t, x, y, sigma_x, sigma_y) ou sigma_* est
        l'ecart-type 1-sigma de la position estimee a cet instant
        (le cone d'incertitude grandit avec l'horizon, c'est normal :
        plus on projette loin, moins on est sur).
        """
        x, P, t = self.x.copy(), self.P.copy(), self.t
        out = []
        n_steps = int(horizon_s / dt)
        for _ in range(n_steps):
            x, P = self._predict_from(x, P, t, t + dt)
            t += dt
            sigma_x = float(np.sqrt(P[0, 0]))
            sigma_y = float(np.sqrt(P[1, 1]))
            out.append((t, float(x[0, 0]), float(x[1, 0]), sigma_x, sigma_y))
        return out
