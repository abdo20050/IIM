import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class VectorizedKalmanTracker:
    """
    A vectorized Kalman Filter tracker for high-performance object tracking.
    Replaces the object-based SortPointTracker for better speed with large N.
    """
    _id_count = 0

    def __init__(self, max_age=5, min_hits=1, distance_threshold=60):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_thresh = distance_threshold
        
        # State vectors (N, 4): [x, y, vx, vy]
        self.X = np.empty((0, 4), dtype=np.float32)
        
        # Covariance matrices (N, 4, 4)
        self.P = np.empty((0, 4, 4), dtype=np.float32)
        
        # Metadata
        self.ids = np.empty((0,), dtype=int)
        self.hits = np.empty((0,), dtype=int)
        self.time_since_update = np.empty((0,), dtype=int)
        self.hit_streak = np.empty((0,), dtype=int)
        
        self.frame_count = 0
        
        # Kalman Parameters
        self.dt = 1.0
        self.F = np.array([[1, 0, 1, 0], 
                           [0, 1, 0, 1], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]], dtype=np.float32)
                           
        # Noise matrices
        self.R = np.eye(2, dtype=np.float32) * 10.0
        self.Q = np.eye(4, dtype=np.float32)
        self.Q[2:, 2:] *= 0.01
        self.Q[:2, :2] *= 0.1
        
        self.init_P = np.eye(4, dtype=np.float32) * 100.0

    def predict(self):
        if len(self.X) == 0:
            return
        
        # X = F @ X.T -> (N, 4)
        self.X = self.X @ self.F.T
        
        # P = F @ P @ F.T + Q
        # Vectorized matrix multiplication: (N, 4, 4)
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.time_since_update += 1

    def update(self, detections):
        """
        Update tracks with new detections.
        detections: (M, 2) array of [x, y]
        """
        self.frame_count += 1
        self.predict()
        
        detections = np.array(detections, dtype=np.float32)
        if detections.ndim == 1:
            detections = detections.reshape(-1, 2)
            
        if len(self.X) == 0:
            self._init_trackers(detections)
            return self._get_results()
            
        if len(detections) == 0:
            self.X[:, 2:] *= 0.9
            return self._get_results()

        # Association
        pred_pos = self.X[:, :2]
        cost_matrix = cdist(pred_pos, detections)
        
        # Use greedy matching if problem size is large (heuristic > 500x500), else Hungarian
        if cost_matrix.size > 250000: 
            matches = self._greedy_match(cost_matrix)
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Filter by threshold
            matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.dist_thresh:
                    matches.append([r, c])
            matches = np.array(matches)

        # Identify unmatched
        unmatched_trks = np.ones(len(self.X), dtype=bool)
        unmatched_dets = np.ones(len(detections), dtype=bool)
        
        if len(matches) > 0:
            unmatched_trks[matches[:, 0]] = False
            unmatched_dets[matches[:, 1]] = False
            
            # Kalman Update for matched
            trks_idx = matches[:, 0]
            dets_idx = matches[:, 1]
            
            Z = detections[dets_idx]
            y = Z - self.X[trks_idx, :2]
            
            # S = H P H.T + R
            # H selects top-left 2x2 of P
            P_sub = self.P[trks_idx]
            S = P_sub[:, :2, :2] + self.R
            
            # K = P H.T S^-1
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
                
            K = P_sub[:, :, :2] @ S_inv
            
            # x_new = x + K y
            ky = (K @ y[:, :, None]).squeeze(-1)
            self.X[trks_idx] += ky
            
            # P_new = (I - K H) P
            KH = K @ self.H[None, :, :]
            I = np.eye(4, dtype=np.float32)
            self.P[trks_idx] = (I - KH) @ P_sub
            
            # Update stats
            self.hits[trks_idx] += 1
            self.hit_streak[trks_idx] += 1
            self.time_since_update[trks_idx] = 0
            
        # Apply friction to unmatched trackers to prevent drift
        if np.any(unmatched_trks):
            self.X[unmatched_trks, 2:] *= 0.9
            
        # Initialize new trackers
        new_det_indices = np.where(unmatched_dets)[0]
        if len(new_det_indices) > 0:
            self._init_trackers(detections[new_det_indices])
            
        # Remove dead trackers
        alive_mask = self.time_since_update <= self.max_age
        self._compress(alive_mask)
        
        return self._get_results()

    def _greedy_match(self, cost_matrix):
        # Sort trackers by their closest detection distance
        mins = np.min(cost_matrix, axis=1)
        sorted_trks = np.argsort(mins)
        
        matches = []
        used_dets = set()
        
        for t in sorted_trks:
            d = np.argmin(cost_matrix[t])
            dist = cost_matrix[t, d]
            if d not in used_dets and dist <= self.dist_thresh:
                matches.append([t, d])
                used_dets.add(d)
        return np.array(matches)

    def _init_trackers(self, detections):
        n = len(detections)
        if n == 0: return
        new_X = np.zeros((n, 4), dtype=np.float32)
        new_X[:, :2] = detections
        new_P = np.tile(self.init_P, (n, 1, 1))
        new_ids = np.arange(VectorizedKalmanTracker._id_count, VectorizedKalmanTracker._id_count + n, dtype=int)
        VectorizedKalmanTracker._id_count += n
        
        self.X = np.concatenate((self.X, new_X))
        self.P = np.concatenate((self.P, new_P))
        self.ids = np.concatenate((self.ids, new_ids))
        self.hits = np.concatenate((self.hits, np.zeros(n, dtype=int)))
        self.hit_streak = np.concatenate((self.hit_streak, np.zeros(n, dtype=int)))
        self.time_since_update = np.concatenate((self.time_since_update, np.zeros(n, dtype=int)))

    def _compress(self, mask):
        self.X = self.X[mask]
        self.P = self.P[mask]
        self.ids = self.ids[mask]
        self.hits = self.hits[mask]
        self.hit_streak = self.hit_streak[mask]
        self.time_since_update = self.time_since_update[mask]

    def _get_results(self):
        mask = (self.hits >= self.min_hits)
        if not np.any(mask):
            return np.empty((0, 4))
        res_X = self.X[mask]
        res_ids = self.ids[mask]
        res_tsu = self.time_since_update[mask]
        is_pred = (res_tsu > 0).astype(float)
        return np.column_stack((res_X[:, 0], res_X[:, 1], res_ids, is_pred))