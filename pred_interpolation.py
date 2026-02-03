import os
import cv2
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from model.locator import Crowd_locator
from collections import OrderedDict
import time
import csv
from misc.compute_metric import eval_metrics
import scipy.spatial
import scipy.optimize
import json
from collections import defaultdict
from fast_tracker import VectorizedKalmanTracker as SortPointTracker


def get_mean_std(dataset):
    if dataset == 'NWPU':
        return ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    if dataset == 'SHHA':
        return ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
    if dataset == 'SHHB':
        return ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    if dataset == 'QNRF':
        return ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])
    if dataset == 'FDST':
        return ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    if dataset == 'JHU':
        return ([0.429683953524, 0.437104910612, 0.421978861094], [0.235549390316, 0.232568427920, 0.2355950474739])
    return ([0.45, 0.45, 0.45], [0.23, 0.23, 0.23])


def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes


def load_gt_data(csv_path):
    """
    Robustly load ground-truth data from a CSV or whitespace-delimited file.
    Returns a dict mapping frame keys to {'num': count, 'points': np.array}.
    """
    gt_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return gt_map

    try:
        with open(csv_path, 'r') as f:
            # Read sample to detect delimiter
            sample = f.read(8192)
            f.seek(0)
            
            # Heuristic for delimiter detection
            if ',' in sample:
                delimiter = ','
            elif '\t' in sample:
                delimiter = '\t'
            else:
                delimiter = None # Will use split() for whitespace

            if delimiter:
                reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
                rows = list(reader)
            else:
                # Fallback for whitespace delimited files
                rows = [line.split() for line in f if line.strip()]
            
            if not rows:
                return gt_map

            # Header detection: if first row's second element is not numeric
            start_idx = 0
            if len(rows[0]) >= 2:
                try:
                    # Remove commas if any (e.g. "1,000")
                    float(str(rows[0][1]).replace(',', ''))
                except (ValueError, IndexError):
                    start_idx = 1

            temp_points = defaultdict(list)
            
            for tokens in rows[start_idx:]:
                tokens = [t.strip() for t in tokens if t and t.strip()]
                if not tokens:
                    continue
                
                key_raw = tokens[0]
                # Try numeric key first
                try:
                    # Handle cases like "1.0" or "001"
                    frame_key = int(float(key_raw))
                except ValueError:
                    # Fallback to filename without extension
                    frame_key = os.path.splitext(os.path.basename(key_raw))[0]

                # Heuristic: if remaining tokens are odd, first is count
                remaining = tokens[1:]
                if len(remaining) % 2 != 0:
                    coord_tokens = remaining[1:]
                else:
                    coord_tokens = remaining

                for i in range(0, len(coord_tokens), 2):
                    if i + 1 >= len(coord_tokens):
                        break
                    try:
                        x = float(coord_tokens[i])
                        y = float(coord_tokens[i+1])
                        temp_points[frame_key].append([x, y])
                    except ValueError:
                        break
            
            for k, pts_list in temp_points.items():
                pts_array = np.array(pts_list)
                gt_map[k] = {
                    'num': len(pts_list),
                    'points': pts_array
                }
                # Also index by string for flexibility
                if isinstance(k, int):
                    gt_map[str(k)] = gt_map[k]

    except Exception as e:
        print(f"Warning: Error loading GT file {csv_path}: {e}")
        
    return gt_map


def remove_close_points(points, min_dist):
    """
    Remove points that are too close to each other (greedy NMS).
    """
    if len(points) < 2:
        return points
    
    dists = scipy.spatial.distance.cdist(points, points)
    n = len(points)
    keep = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not keep[i]:
            continue
        mask = dists[i] < min_dist
        mask[i] = False 
        keep[mask] = False
        
    return points[keep]


def interpolate_points(prev_points, next_points, num_frames_between):
    """
    Interpolate points between two frames using Hungarian matching.
    Returns a list of np.arrays for the intermediate frames.
    """
    if num_frames_between == 0:
        return []
        
    interpolated = [np.empty((0, 2)) for _ in range(num_frames_between)]
    
    if prev_points.size == 0 or next_points.size == 0:
        return interpolated
        
    dists = scipy.spatial.distance.cdist(prev_points, next_points)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dists)
    
    # Filter matches with a loose threshold to avoid cross-screen interpolation
    # Assuming 1080p/4k, 300px is reasonable for movement between skipped frames
    valid_matches = []
    for r, c in zip(row_ind, col_ind):
        if dists[r, c] < 300: 
            valid_matches.append((r, c))
            
    for i in range(num_frames_between):
        alpha = (i + 1) / (num_frames_between + 1)
        pts = []
        for r, c in valid_matches:
            pts.append(prev_points[r] * (1 - alpha) + next_points[c] * alpha)
        if pts:
            interpolated[i] = np.array(pts)
            
    return interpolated


def frame_predict(net, img_bgr, img_transform, slice_h=512, slice_w=1024, device='cuda'):
    # Convert BGR->RGB PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    img_t = img_transform(pil)[None, :, :, :]

    # --- CHANGE 2: Use inference_mode (Faster than no_grad) ---
    with torch.inference_mode():
        # --- CHANGE 3: Remove 'Variable' and cast to Half ---
        # Variable is deprecated. Just use .to()
        # .half() is REQUIRED if you did net.half() in main()
        img_var = img_t.to(device).half() 
        
        b, c, h, w = img_var.shape
        
        # ... (Rest of logic remains the same) ...
        
        # NOTE: When running the loop for slices:
        # crop_imgs.append(img_var[:, :, h_start:h_end, w_start:w_end])
        # Ensure crop_imgs is processed as half automatically since img_var is half.

    # with torch.no_grad():
    #     img_var = Variable(img_t).to(device)
    #     b, c, h, w = img_var.shape
        # tiling logic similar to test.py
        if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
            [pred_threshold, pred_map, __] = [i.cpu() for i in net(img_var, mask_gt=None, mode='val')]
        else:
            if h % 16 != 0:
                pad_dims = (0, 0, 0, 16 - h % 16)
                h = (h // 16 + 1) * 16
                img_var = F.pad(img_var, pad_dims, "constant")

            if w % 16 != 0:
                pad_dims = (0, 16 - w % 16, 0, 0)
                w = (w // 16 + 1) * 16
                img_var = F.pad(img_var, pad_dims, "constant")

            crop_imgs, crop_masks = [], []
            for i in range(0, h, slice_h):
                h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                for j in range(0, w, slice_w):
                    w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                    crop_imgs.append(img_var[:, :, h_start:h_end, w_start:w_end])
                    mask = torch.zeros(1, 1, img_var.size(2), img_var.size(3)).cpu()
                    mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                    crop_masks.append(mask)

            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_masks = torch.cat(crop_masks, dim=0)

            crop_preds, crop_thresholds = [], []
            nz, period = crop_imgs.size(0), 4
            for i in range(0, nz, period):
                [crop_threshold, crop_pred, __] = [i.cpu() for i in net(crop_imgs[i:min(nz, i+period)].to(device), mask_gt=None, mode='val')]
                crop_preds.append(crop_pred)
                crop_thresholds.append(crop_threshold)

            crop_preds = torch.cat(crop_preds, dim=0)
            crop_thresholds = torch.cat(crop_thresholds, dim=0)

            idx = 0
            pred_map = torch.zeros(1, 1, h, w).cpu()
            pred_threshold = torch.zeros(1, 1, h, w).cpu().float()
            for i in range(0, h, slice_h):
                h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                for j in range(0, w, slice_w):
                    w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                    pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                    pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                    idx += 1
            mask = crop_masks.sum(dim=0)
            pred_map = (pred_map / mask)
            pred_threshold = (pred_threshold / mask)

        a = torch.ones_like(pred_map)
        b = torch.zeros_like(pred_map)
        binar_map = torch.where(pred_map >= pred_threshold, a, b)

        pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())
        return pred_data, boxes, pred_map.cpu().numpy(), pred_threshold.cpu().numpy()


def process_frame(frame, frame_num, points, boxes, tracker, gt_map, args, count_errors, stats_raw, out, dt=0.0):
    # update tracker and draw points (with persistent ID)
    detections = np.array(points) if len(points) > 0 else np.empty((0, 2))
    
    tracked = tracker.update(detections)
    
    # tracked rows: [x, y, id, is_predicted]
    if tracked.size > 0:
        for row in tracked:
            if row[3] > 0 and not args.show_ghosts:
                continue
            x, y = int(row[0]), int(row[1])
            tid = int(row[2])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(frame, str(tid), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw boxes only if they exist (processed frames)
    for bb in boxes:
        x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.putText(frame, f'Count: {len(points)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # draw ground-truth circles for visual inspection (radius = match_dist)
    if args.gt is not None:
        gt_vis = gt_map.get(frame_num, gt_map.get(str(frame_num), gt_map.get(os.path.splitext(os.path.basename(args.input))[0], None)))
        if gt_vis is not None and 'points' in gt_vis and getattr(gt_vis['points'], 'size', 0) > 0:
            rr = int(round(args.match_dist))
            pts = np.array(gt_vis['points'])
            if pts.ndim == 1:
                pts = pts.reshape(1, 2)
            if args.verbose:
                print(f"GT points (frame {frame_num}): {len(pts)}")
            for gx, gy in pts:
                try:
                    cv2.circle(frame, (int(gx), int(gy)), rr, (255, 255, 255), 1)
                except Exception:
                    continue

    # compute frame metrics if ground-truth provided
    if args.gt is not None:
        key = frame_num
        gt = gt_map.get(key, gt_map.get(str(key), gt_map.get(os.path.splitext(os.path.basename(args.input))[0], None)))
        if gt is not None:
            gt_count = float(gt['num'])
            pred_count = float(len(points))
            se = (gt_count - pred_count) * (gt_count - pred_count)
            ae = abs(gt_count - pred_count)
            count_errors['mse_sum'] += se
            count_errors['mae_sum'] += ae
            count_errors['n'] += 1

            try:
                if tracked is not None and tracked.size > 0:
                    preds = np.array([row[:2] for row in tracked])
                else:
                    preds = np.array(points) if len(points) > 0 else np.empty((0, 2))
            except Exception:
                preds = np.array(points) if len(points) > 0 else np.empty((0, 2))

            gts = np.array(gt['points']) if gt['points'].size > 0 else np.empty((0, 2))
            if gts.ndim == 1 and gts.size == 2:
                gts = gts.reshape(1, 2)

            def compute_metrics(preds, gts, dist_threshold=args.match_dist):
                tp = fp = fn = 0
                total_error = 0.0
                matches = 0
                if preds.size == 0 and gts.size == 0:
                    return {'tp': 0, 'fp': 0, 'fn': 0, 'error': 0.0, 'matches': 0}
                if preds.size == 0:
                    return {'tp': 0, 'fp': 0, 'fn': len(gts), 'error': 0.0, 'matches': 0}
                if gts.size == 0:
                    return {'tp': 0, 'fp': len(preds), 'fn': 0, 'error': 0.0, 'matches': 0}

                d_matrix = scipy.spatial.distance.cdist(preds, gts, metric='euclidean')
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(d_matrix)

                used_preds = set()
                used_gts = set()
                for r, c in zip(row_ind, col_ind):
                    dist = d_matrix[r, c]
                    if dist <= dist_threshold:
                        tp += 1
                        total_error += dist
                        used_preds.add(r)
                        used_gts.add(c)

                fp = len(preds) - len(used_preds)
                fn = len(gts) - len(used_gts)
                matches = tp
                return {'tp': tp, 'fp': fp, 'fn': fn, 'error': total_error, 'matches': matches}

            m = compute_metrics(preds, gts, dist_threshold=args.match_dist)
            stats_raw['tp'] += m['tp']
            stats_raw['fp'] += m['fp']
            stats_raw['fn'] += m['fn']
            stats_raw['error'] += m['error']
            stats_raw['matches'] += m['matches']

            p_r = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
            r_r = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
            f1_r = 2 * (p_r * r_r) / (p_r + r_r) if (p_r + r_r) > 0 else 0
            mae_r = m['error'] / m['matches'] if m['matches'] > 0 else 0
            stats_raw['frame_data'].append({'frame': frame_num, 'p': p_r, 'r': r_r, 'f1': f1_r, 'mae': mae_r, 'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'], 'time': dt, 'gt_count': int(gt_count), 'pred_count': int(pred_count)})

            print(f"Frame {frame_num:4d} | TP:{m['tp']:3d} FP:{m['fp']:3d} FN:{m['fn']:3d} | Prec:{p_r:.3f} Rec:{r_r:.3f} F1:{f1_r:.3f} | MAE:{mae_r:.2f} | t:{dt:.4f}s")

    # write frame to output (if writer initialized)
    try:
        if out is not None:
            out.write(frame)
    except Exception:
        pass

    if args.verbose and dt > 0:
        print(f'Frame {frame_num} time: {dt:.4f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input video path')
    parser.add_argument('--output', '-o', required=True, help='output video path')
    parser.add_argument('--model', '-m', required=True, help='model .pth path')
    parser.add_argument('--dataset', default='JHU', help='dataset mean/std (JHU,NWPU,...)')
    parser.add_argument('--net', default='HR_Net', help='network name (HR_Net or VGG16_FPN)')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES id(s)')
    parser.add_argument('--gt', default=None, help='optional ground-truth file (same format as test output: filename count x1 y1 x2 y2 ...)')
    parser.add_argument('--match_dist', type=float, default=12.0, help='distance threshold (pixels) for matching predicted to GT')
    parser.add_argument('--out_dir', default=None, help='directory to save JSON/CSV reports (optional)')
    parser.add_argument('--merge_dist', type=float, default=8.0, help='distance threshold (pixels) to merge close points (double detections)')
    parser.add_argument('--track_max_age', type=int, default=50, help='Max frames to keep a track alive without detection')
    parser.add_argument('--track_dist_thresh', type=float, default=50.0, help='Distance threshold for tracking association')
    parser.add_argument('--show_ghosts', action='store_true', help='Visualize predicted points (ghosts) when detection is lost')
    parser.add_argument('--skip_frames', type=int, default=1, help='Process every Nth frame (default 1 = all frames)')
    parser.add_argument('--verbose', action='store_true', help='print per-frame timing')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    mean_std = get_mean_std(args.dataset)
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    net = Crowd_locator(args.net, args.gpu, pretrained=True)
    net.to(device)
    state = torch.load(args.model, map_location=device)
    # support both single-GPU and state dicts wrapped with additional keys
    if isinstance(state, dict) and ('net' in state or 'state_dict' in state):
        # try common containers
        sd = state.get('net', state.get('state_dict', state))
    else:
        sd = state

    # strip module. if needed
    new_state = OrderedDict()
    for k, v in sd.items():
        name = k.replace('module.', '')
        new_state[name] = v
    try:
        net.load_state_dict(new_state)
    except Exception:
        # fallback: try load directly
        try:
            net.load_state_dict(sd)
        except Exception as e:
            print('Model loading failed:', e)
            return

    net.eval()    
    
    # --- CHANGE 1: Optimize the Model ---
    # 1. Convert to Half Precision (Speedup + Less VRAM)
    net.half() 
    
    # 2. Compile (PyTorch 2.0+ only)
    # This fuses layers for faster inference
    try:
        net = torch.compile(net, mode="reduce-overhead")
        print("Model compiled successfully with torch.compile!")
    except Exception as e:
        print(f"Warning: Could not compile model (ignore if on older PyTorch): {e}")

    # load ground-truth mapping if provided
    gt_map = load_gt_data(args.gt)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print('Failed to open input:', args.input)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # OpenH264 requires even dimensions (multiples of 2)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = orig_w & ~1
    height = orig_h & ~1

    # Try H.264 (OpenH264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Warning: 'avc1' (H.264) failed. This usually means libopenh264.so.4 is missing or incompatible.")
        print("Attempting 'X264' as alternative...")
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Warning: H.264 failed. Trying 'mp4v' (MPEG-4) fallback...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
    if not out.isOpened():
        print(f"Error: VideoWriter failed to initialize. Check your FFmpeg/OpenH264 installation.")
        cap.release()
        return

    frame_idx = 0
    times = []
    # counters for metrics (raw predictions)
    count_errors = {'mae_sum': 0.0, 'mse_sum': 0.0, 'n': 0}
    stats_raw = {'tp': 0, 'fp': 0, 'fn': 0, 'error': 0.0, 'matches': 0, 'frame_data': []}

    # instantiate tracker for persistent IDs
    tracker = SortPointTracker(max_age=args.track_max_age, min_hits=1, distance_threshold=args.track_dist_thresh)
    
    print('Processing video...')
    prev_points = None
    prev_boxes = []
    frame_buffer = [] # Stores tuple: (frame_image, frame_index)

    print('Processing video...')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame should be a "Keyframe" (Detection Frame)
        is_keyframe = (frame_idx % args.skip_frames == 0)

        if is_keyframe:
            # 1. Run Detection (Heavy computation)
            if device == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # Predict
            pred_data, boxes, _, _ = frame_predict(net, frame, img_transform, device=device)
            
            if device == 'cuda': torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt) # Only log time for actual detection frames

            # Extract points
            curr_points = pred_data['points'] if pred_data['num'] > 0 else np.empty((0, 2))
            if len(curr_points) > 0 and args.merge_dist > 0:
                curr_points = remove_close_points(curr_points, args.merge_dist)

            # 2. Interpolate Missing Frames (The Gap)
            # If we have a buffer of skipped frames, fill them in now that we have the start (prev) and end (curr)
            if frame_buffer and prev_points is not None:
                # Generate points for all frames in the buffer
                interp_list = interpolate_points(prev_points, curr_points, len(frame_buffer))
                
                for i, (buf_frame, buf_idx) in enumerate(frame_buffer):
                    # Use interpolated points. 
                    # Note: We pass empty boxes [] because we don't have heatmap data for skipped frames
                    process_frame(buf_frame, buf_idx, interp_list[i], [], tracker, gt_map, args, count_errors, stats_raw, out, dt=0.0)

                # Clear the buffer after processing
                frame_buffer = []

            # 3. Process the Current Keyframe
            process_frame(frame, frame_idx, curr_points, boxes, tracker, gt_map, args, count_errors, stats_raw, out, dt)

            # 4. Update History
            prev_points = curr_points
            prev_boxes = boxes # Optional, if you wanted to interpolate boxes too

        else:
            # This is a SKIPPED frame. 
            # We buffer it because we cannot process it until we know the NEXT detection result.
            frame_buffer.append((frame, frame_idx))
            
            # Edge case: If start of video is not 0 (unlikely) or detection failed earlier
            if prev_points is None:
                prev_points = np.empty((0, 2))

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f'Processed {frame_idx} frames...')

    # 5. Cleanup: Handle any remaining buffered frames (e.g., video ended before next keyframe)
    if frame_buffer and prev_points is not None:
         for i, (buf_frame, buf_idx) in enumerate(frame_buffer):
             # We can't interpolate forward, so just hold the last known points
             process_frame(buf_frame, buf_idx, prev_points, [], tracker, gt_map, args, count_errors, stats_raw, out, dt=0.0)

    cap.release()
    out.release()
    # summary timings
    if len(times) > 1:
        times = times[1:] # exclude first frame
        avg_t = float(np.mean(times))
        std_t = float(np.std(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
        print(f'Frame processing time: avg={avg_t:.4f}s, std={std_t:.4f}s, min={min_t:.4f}s, max={max_t:.4f}s ({1.0/avg_t:.2f} FPS)')
    else:
        avg_t = std_t = min_t = max_t = 0.0
        print('No frames processed.')

    # summary metrics (counts)
    if count_errors['n'] > 0:
        mse = count_errors['mse_sum'] / count_errors['n']
        mae = count_errors['mae_sum'] / count_errors['n']
        print(f'Count MAE: {mae:.4f}, Count MSE: {mse:.4f}')

    # compute micro/macro for raw detection stats
    if len(stats_raw['frame_data']) > 0:
        tot_tp = stats_raw['tp']
        tot_fp = stats_raw['fp']
        tot_fn = stats_raw['fn']
        mic_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
        mic_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
        mic_f1 = 2 * mic_p * mic_r / (mic_p + mic_r) if (mic_p + mic_r) > 0 else 0
        mic_mae = stats_raw['error'] / stats_raw['matches'] if stats_raw['matches'] > 0 else 0

        # macro (per-frame average)
        fdata = stats_raw['frame_data']
        mac_p = sum(x['p'] for x in fdata) / len(fdata)
        mac_r = sum(x['r'] for x in fdata) / len(fdata)
        mac_f1 = sum(x['f1'] for x in fdata) / len(fdata)
        mac_mae = sum(x['mae'] for x in fdata) / len(fdata)

        print(f'Detection (Raw) Micro - Precision: {mic_p:.4f}, Recall: {mic_r:.4f}, F1: {mic_f1:.4f}, MAE(px): {mic_mae:.4f}')
        print(f'Detection (Raw) Macro - Precision: {mac_p:.4f}, Recall: {mac_r:.4f}, F1: {mac_f1:.4f}, MAE(px): {mac_mae:.4f}')
        
        # export reports if requested
        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(args.input))[0]
        json_path = os.path.join(out_dir, f"{base_name}_metrics.json")
        csv_path = os.path.join(out_dir, f"{base_name}_frame_metrics.csv")
        summary_csv = os.path.join(out_dir, f"{base_name}_summary.csv")

        # JSON report
        def round3(v): return round(v, 3)
        json_report = {
            'video_file': os.path.basename(args.input),
            'total_frames': len(stats_raw['frame_data']),
            'avg_frame_time_s': round3(avg_t),
            'std_frame_time_s': round3(std_t),
            'min_frame_time_s': round3(min_t),
            'max_frame_time_s': round3(max_t),
            'raw': {
                'micro': {'precision': round3(mic_p), 'recall': round3(mic_r), 'f1': round3(mic_f1), 'mae_px': round3(mic_mae)},
                'macro': {'precision': round3(mac_p), 'recall': round3(mac_r), 'f1': round3(mac_f1), 'mae_px': round3(mac_mae)}
            }
        }
        try:
            with open(json_path, 'w') as jf:
                json.dump(json_report, jf, indent=4)
            print('Exported JSON report to', json_path)
        except Exception as e:
            print('Failed to write JSON report:', e)

        # per-frame CSV
        try:
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(['frame', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mae_px', 'time_s'])
                for idx, row in enumerate(stats_raw['frame_data']):
                    writer.writerow([idx + 1, row.get('gt_count', 0), row.get('pred_count', 0), row.get('tp', 0), row.get('fp', 0), row.get('fn', 0), round3(row.get('p', 0)), round3(row.get('r', 0)), round3(row.get('f1', 0)), round3(row.get('mae', 0)), round3(row.get('time', 0))])
            print('Exported per-frame CSV to', csv_path)
        except Exception as e:
            print('Failed to write per-frame CSV:', e)

        # summary CSV
        try:
            with open(summary_csv, 'w', newline='') as sf:
                writer = csv.writer(sf)
                writer.writerow(['metric', 'aggregation', 'value'])
                writer.writerow(['precision', 'micro', round3(mic_p)])
                writer.writerow(['recall', 'micro', round3(mic_r)])
                writer.writerow(['f1', 'micro', round3(mic_f1)])
                writer.writerow(['mae_px', 'micro', round3(mic_mae)])
                writer.writerow(['precision', 'macro', round3(mac_p)])
                writer.writerow(['recall', 'macro', round3(mac_r)])
                writer.writerow(['f1', 'macro', round3(mac_f1)])
                writer.writerow(['mae_px', 'macro', round3(mac_mae)])
            print('Exported summary CSV to', summary_csv)
        except Exception as e:
            print('Failed to write summary CSV:', e)

    print('Finished. Output saved to', args.output)


if __name__ == '__main__':
    main()
