# FreqTrack: A Frequency-Enhanced Spatiotemporal Network
# Implementation of Phase II: Frequency-Aware Tracking Scheme

from collections import deque
import numpy as np
import cv2

# Try importing standard tracking utilities (e.g., from YOLOX/ByteTrack)
try:
    from .basetrack import TrackState
    from .byte_tracker import BYTETracker, STrack
    from .utils import matching
    from .utils.kalman_filter import KalmanFilterXYWH
except ImportError:
    try:
        from basetrack import TrackState
        from byte_tracker import BYTETracker, STrack
        import utils.matching as matching
        from utils.kalman_filter import KalmanFilterXYWH
    except ImportError:
        print("Error: Missing tracking dependency files. Please ensure ByteTrack/BoTSORT utils are in your path.")
        raise


class SpectralGMC:
    """
    Spectral Global Motion Compensation (SGMC).
    Estimates global camera motion between consecutive frames using Phase Correlation 
    in the frequency domain. Highly robust to motion blur and textureless backgrounds.
    """
    def __init__(self, downscale=2, response_thresh=0.05):
        self.prev_gray = None
        self.downscale = downscale
        self.response_thresh = response_thresh

    def apply(self, raw_frame, detections=None):
        if raw_frame.ndim > 2:
            curr_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = raw_frame

        height, width = curr_gray.shape
        curr_gray_resized = cv2.resize(curr_gray, (width // self.downscale, height // self.downscale))
        H = np.eye(2, 3)

        if self.prev_gray is not None:
            (dx, dy), response = cv2.phaseCorrelate(
                self.prev_gray.astype(np.float32),
                curr_gray_resized.astype(np.float32)
            )
            if response > self.response_thresh:
                dx *= self.downscale
                dy *= self.downscale
                H[0, 2] = dx
                H[1, 2] = dy

        self.prev_gray = curr_gray_resized
        return H


class BOTrack(STrack):
    """
    Tracking state management integrated with Fourier Fingerprint features.
    """
    shared_kalman = KalmanFilterXYWH() 

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        _element = np.append(np.array(tlwh), float(score))
        super().__init__(_element, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def convert_coords(self, tlwh):
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert tlwh bounding box to xywh format (cx, cy, w, h)"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov


class FreqTracker(BYTETracker):
    """
    FreqTrack: Frequency-Aware Multi-Object Tracker.
    Integrates Spectral Global Motion Compensation and Spectral Texture Consistency.
    """
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)

        self.lambda_iou = getattr(args, 'lambda_iou', 0.2)
        self.lambda_freq = getattr(args, 'lambda_freq', 0.3)  # Optimal value derived from ablation studies

        print(f"Initializing FreqTracker with lambda_freq = {self.lambda_freq}")
        self.proximity_thresh = getattr(args, 'proximity_thresh', 0.5)
        self.appearance_thresh = getattr(args, 'appearance_thresh', 0.25)

        self.encoder = None
        self.gmc = SpectralGMC(downscale=2, response_thresh=0.05)

    def init_track(self, dets, scores, cls, features=None):
        if len(dets) == 0:
            return []
        if features is not None and len(features) > 0:
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features)]
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]

    def get_dists(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))

        dists_iou = matching.iou_distance(tracks, detections)
        dists_freq = matching.embedding_distance(tracks, detections)

        if dists_freq.size == 0 or dists_freq.shape != dists_iou.shape:
            return dists_iou

        dists = self.lambda_iou * dists_iou + self.lambda_freq * dists_freq

        dists_mask = (dists_iou > self.proximity_thresh) | (dists_freq > self.appearance_thresh)
        dists[dists_mask] = np.inf
        dists = matching.fuse_score(dists, detections)
        return dists

    def update(self, results, img=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(results) > 0:
            dets = results[:, :4]
            scores = results[:, 4]
            cls = results[:, 5]
            features = results[:, 6:]
        else:
            dets = np.array([])
            scores = np.array([])
            cls = np.array([])
            features = np.array([])

        if img is not None:
            warp_matrix = self.gmc.apply(img, dets)
            STrack.multi_gmc(self.tracked_stracks, warp_matrix)
            STrack.multi_gmc(self.lost_stracks, warp_matrix)

        remain_inds = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        dets_first = dets[remain_inds]
        scores_first = scores[remain_inds]
        cls_first = cls[remain_inds]
        feats_first = features[remain_inds] if len(features) > 0 else None

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        scores_second = scores[inds_second]
        cls_second = cls[inds_second]
        feats_second = None

        detections = self.init_track(dets_first, scores_first, cls_first, feats_first)
        detections_second = self.init_track(dets_second, scores_second, cls_second, feats_second)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        if len(unconfirmed) > 0:
            self.multi_predict(unconfirmed)

        # 1. First Association
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 2. Second Association
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 3. Unconfirmed Association
        detections_rem = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections_rem)
        matches, u_unconfirmed, u_detection_rem = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections_rem[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 4. Init New Tracks
        for inew in u_detection_rem:
            track = detections_rem[inew]
            if track.score < self.args.track_new_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 5. Remove Lost Tracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = pairs
    for p, q in zip(dupa, dupb):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            stracksb[q].mark_removed()
        else:
            stracksa[p].mark_removed()
    return [t for t in stracksa if t.state != TrackState.Removed], [t for t in stracksb if t.state != TrackState.Removed]