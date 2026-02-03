import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class KalmanPointTracker(object):
    count = 0
    def __init__(self, initial_point):
        self.kf = KalmanFilter(dim_x=4, dim_z=2) 
        
        # State: [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # --- TUNING FOR MOMENTUM ---
        
        # P: Covariance. 
        self.kf.P *= 100.0 

        # R: Measurement Noise. 
        # Kept high (10.0) to smooth out path jitter. 
        # This ensures the velocity vector points in the general direction of movement,
        # rather than snapping to the last noisy pixel.
        self.kf.R *= 10.0   
        
        # Q: Process Noise. 
        # Very low velocity noise enforces "Inertia" (keep moving in same line).
        self.kf.Q[-1, -1] *= 0.01 
        self.kf.Q[-2, -2] *= 0.01 
        self.kf.Q[:2, :2] *= 0.1   

        self.kf.x[:2] = initial_point.reshape((2, 1))
        
        self.time_since_update = 0
        self.id = KalmanPointTracker.count
        KalmanPointTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, point):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(point)

        # --- VELOCITY WARM-UP (SOFTENED) ---
        # Previously was 0.5 (too harsh). Now 0.9.
        # This prevents noise from launching a point, but allows real movement 
        # to build up speed quickly.
        if self.hits < 3:
            self.kf.x[2] *= 0.9 
            self.kf.x[3] *= 0.9 

    def predict(self):
        # 1. Standard Prediction
        self.kf.predict()
        
        # --- VELOCITY CLAMPING ---
        # Cap max speed to prevent "teleporting"
        max_speed = 120
        self.kf.x[2] = np.clip(self.kf.x[2], -max_speed, max_speed)
        self.kf.x[3] = np.clip(self.kf.x[3], -max_speed, max_speed)

        # --- LOW FRICTION (GLIDE) ---
        # Previously 0.9 (brake). Now 0.98 (glide).
        # 0.98^30 ~= 0.55 (Retains 55% speed after 1 second)
        if self.time_since_update > 0:
            speed = np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)
            decay = 0.98
            
            # Only apply stronger braking if it's really flying (>15px/frame)
            if speed > 15: 
                decay = 0.95
            
            self.kf.x[2] *= decay
            self.kf.x[3] *= decay
            
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:2].reshape((1, 2))

    def get_state(self):
        return self.kf.x[:2].reshape((1, 2))

class SortPointTracker(object):
    def __init__(self, max_age=5, min_hits=1, distance_threshold=60):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        
        # 1. Predict
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trks[t] = pos.flatten()
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # 2. Associate
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)

        # 3. Update
        for t, d in matched:
            self.trackers[t].update(detections[d, :])

        # 4. Create New
        for i in unmatched_dets:
            trk = KalmanPointTracker(detections[i, :])
            self.trackers.append(trk)

        # 5. Output Logic
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Output if active.
            # Logic: If it's a "ghost" (time_since_update > 0), we assume it's valid
            # as long as it hasn't exceeded max_age.
            if (trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits):
                is_predicted = 1 if trk.time_since_update > 0 else 0
                ret.append(np.concatenate((d.flatten(), [trk.id, is_predicted])).reshape(1, -1)) 
            
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 4))

    def associate(self, detections, trackers):
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        # Vectorized distance calculation (Massive speedup for dense crowds)
        cost_matrix = cdist(trackers, detections).astype(np.float32)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in col_ind):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in row_ind):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if (cost_matrix[m[0], m[1]] > self.distance_threshold):
                unmatched_detections.append(m[1])
                unmatched_trackers.append(m[0])
            else:
                matches.append(m.reshape(1, 2))
        
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)