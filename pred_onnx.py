import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import time
import csv
import json
from collections import OrderedDict, defaultdict
import scipy.spatial
import scipy.optimize
from kalman_tracker import SortPointTracker

def prepare_quantized_model(model_path, precision):
    """
    Performs on-the-fly quantization of an FP32 ONNX model.
    Returns the path to the quantized model.
    """
    if precision == 'fp32':
        return model_path

    base, ext = os.path.splitext(model_path)
    output_path = f"{base}_{precision}{ext}"

    if os.path.exists(output_path):
        print(f"[*] Using existing {precision} model: {output_path}")
        return output_path

    print(f"[*] Quantizing FP32 model to {precision}...")
    
    if precision == 'fp16':
        try:
            import onnx
            from onnxconverter_common import float16
            model = onnx.load(model_path)
            model_fp16 = float16.convert_float16(model)
            onnx.save(model_fp16, output_path)
        except ImportError:
            print("[!] Error: 'onnxconverter-common' is required for FP16 conversion.")
            return model_path

    elif precision == 'int8':
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=model_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8
            )
        except ImportError:
            print("[!] Error: 'onnxruntime' quantization tools not found.")
            return model_path

    elif precision == 'int4':
        try:
            # Weight-only quantization for 4-bit (requires recent ORT)
            from onnxruntime.quantization import MatMul4BitsQuantizer
            import onnx
            model = onnx.load(model_path)
            # block_size=32 is a common default for 4-bit quantization
            quantizer = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True)
            quantizer.process()
            onnx.save(model, output_path)
        except (ImportError, AttributeError):
            print("[!] Error: INT4 quantization requires a recent version of onnxruntime. Falling back to INT8.")
            return prepare_quantized_model(model_path, 'int8')

    print(f"[+] Quantization complete: {output_path}")
    return output_path

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

def preprocess(img_bgr, mean, std, precision='fp32'):
    # BGR to RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Optimized Normalization using OpenCV
    img = img.astype(np.float32)
    img -= (mean * 255.0).astype(np.float32)
    img /= (std * 255.0).astype(np.float32)
    
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    
    if precision == 'fp16':
        return img.astype(np.float16)
    return img.astype(np.float32)

def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)
    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    return {'num': len(points), 'points': points}, boxes

def load_gt_data(csv_path):
    gt_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return gt_map
    try:
        with open(csv_path, 'r') as f:
            rows = [line.split() for line in f if line.strip()]
            temp_points = defaultdict(list)
            for tokens in rows:
                try:
                    frame_key = int(float(tokens[0]))
                except:
                    frame_key = os.path.splitext(os.path.basename(tokens[0]))[0]
                coord_tokens = tokens[2:] if len(tokens[1:]) % 2 != 0 else tokens[1:]
                for i in range(0, len(coord_tokens), 2):
                    temp_points[frame_key].append([float(coord_tokens[i]), float(coord_tokens[i+1])])
            for k, pts_list in temp_points.items():
                gt_map[k] = {'num': len(pts_list), 'points': np.array(pts_list)}
                if isinstance(k, int): gt_map[str(k)] = gt_map[k]
    except Exception as e:
        print(f"Warning: Error loading GT: {e}")
    return gt_map

def run_tiled_inference(session, img_np, slice_h=512, slice_w=1024, batch_size=4, verbose=False):
    b, c, h, w = img_np.shape
    orig_h, orig_w = h, w
    
    # --- OPTIMIZATION: Smart Tiling ---
    # If image is slightly larger than slice, just process it in one go to avoid tile overhead
    if h < slice_h * 1.2 and w < slice_w * 1.2:
        if verbose: print(f"[*] Image {h}x{w} is small enough; skipping tiling for speed.")
        pad_h = (16 - h % 16) if h % 16 != 0 else 0
        pad_w = (16 - w % 16) if w % 16 != 0 else 0
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
        outputs = session.run(None, {"input": img_np})
        pred_thresh = outputs[0][:, :, :orig_h, :orig_w]
        pred_map = outputs[1][:, :, :orig_h, :orig_w]
        return (pred_map >= pred_thresh).astype(np.float32), pred_map, pred_thresh

    pad_h = (16 - h % 16) if h % 16 != 0 else 0
    pad_w = (16 - w % 16) if w % 16 != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((b, c, orig_h + pad_h, orig_w + pad_w), dtype=img_np.dtype)
        padded[:, :, :orig_h, :orig_w] = img_np
        img_np = padded
        _, _, h, w = img_np.shape

    pred_map_full = np.zeros((1, 1, h, w), dtype=np.float32)
    pred_thresh_full = np.zeros((1, 1, h, w), dtype=np.float32)
    count_mask = np.zeros((1, 1, h, w), dtype=np.float32)

    crops = []
    coords = []

    if verbose:
        print(f"[*] Tiling: Image {orig_h}x{orig_w} padded to {h}x{w}. Processing in {slice_h}x{slice_w} slices.")

    # 1. Collect all tiles
    for i in range(0, h, slice_h):
        h_start = max(min(h - slice_h, i), 0)
        h_end = h_start + slice_h
        for j in range(0, w, slice_w):
            w_start = max(min(w - slice_w, j), 0)
            w_end = w_start + slice_w
            crops.append(img_np[:, :, h_start:h_end, w_start:w_end])
            coords.append((h_start, h_end, w_start, w_end))

    # 2. Run inference in batches with IOBinding (if possible)
    # We use standard run here for compatibility, but batching is the key
    for k in range(0, len(crops), batch_size):
        batch = np.ascontiguousarray(np.vstack(crops[k:k+batch_size]))
        
        # --- OPTIMIZATION: IOBinding ---
        # This avoids copying the input/output between CPU/GPU multiple times
        io_binding = session.io_binding()
        device_id = 0 # Default
        
        # Bind Input
        input_ort = ort.OrtValue.ortvalue_from_numpy(batch, 'cuda', device_id)
        io_binding.bind_ortvalue_input('input', input_ort)
        
        # Bind Outputs (Pre-allocate on GPU to avoid sync)
        # We don't know exact shapes for dynamic, so we let ORT allocate on GPU
        io_binding.bind_output('pred_threshold', 'cuda', device_id)
        io_binding.bind_output('pred_map', 'cuda', device_id)
        io_binding.bind_output('extra_info', 'cuda', device_id)
        
        session.run_with_iobinding(io_binding)
        
        # Retrieve only what we need back to CPU
        outputs = io_binding.copy_outputs_to_cpu()
        
        for idx, (h_s, h_e, w_s, w_e) in enumerate(coords[k:k+batch_size]):
            # outputs[0] is pred_threshold, outputs[1] is pred_map
            pred_thresh_full[:, :, h_s:h_e, w_s:w_e] += outputs[0][idx:idx+1]
            pred_map_full[:, :, h_s:h_e, w_s:w_e] += outputs[1][idx:idx+1]
            count_mask[:, :, h_s:h_e, w_s:w_e] += 1.0

    pred_map_full /= count_mask
    pred_thresh_full /= count_mask
    
    # Crop back to original size
    pred_map_full = pred_map_full[:, :, :orig_h, :orig_w]
    pred_thresh_full = pred_thresh_full[:, :, :orig_h, :orig_w]
    
    binar_map = (pred_map_full >= pred_thresh_full).astype(np.float32)
    return binar_map, pred_map_full, pred_thresh_full

def main():
    parser = argparse.ArgumentParser(description="IIM ONNX Inference Script")
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--model', '-m', required=True, help='ONNX model path')
    parser.add_argument('--dataset', default='JHU', help='Dataset for mean/std')
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'int8', 'int4'], 
                        help='Inference precision. Note: int8/int4 requires a pre-quantized ONNX model.')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--gt', default=None, help='Optional GT file')
    parser.add_argument('--match_dist', type=float, default=12.0, help='Matching distance')
    parser.add_argument('--out_dir', default=None, help='Report directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for tiled inference')
    args = parser.parse_args()

    # 1. Initialize ONNX Runtime Session
    device_id = int(args.gpu) if args.gpu.isdigit() else 0
    
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
    
    # If using TensorRT for potential INT8/FP16 speedup
    if args.precision in ['int8', 'fp16']:
        providers.insert(0, ('TensorrtExecutionProvider', {
            'device_id': device_id,
            'trt_fp16_enable': True if args.precision == 'fp16' else False,
            'trt_int8_enable': True if args.precision == 'int8' else False,
        }))

    # 2. Handle Quantization in-code
    target_model = prepare_quantized_model(args.model, args.precision)

    print(f"[*] Loading ONNX model from {target_model}...")
    session = ort.InferenceSession(target_model, providers=providers)
    
    if args.verbose:
        active_providers = session.get_providers()
        # CRITICAL: Check if requested acceleration is active
        if args.precision in ['int8', 'int4'] and 'TensorrtExecutionProvider' not in active_providers:
            print(f"\n[!!!] WARNING: {args.precision.upper()} requested but TensorrtExecutionProvider is NOT active.")
            print("[!!!] INT8/INT4 on CUDAExecutionProvider is extremely slow. Expect 'seconds per frame'.")
            print("[!!!] Fix: Install TensorRT and set LD_LIBRARY_PATH.\n")
        elif 'CUDAExecutionProvider' not in active_providers:
            print("\n[!!!] WARNING: CUDAExecutionProvider NOT FOUND. Falling back to CPU.")
            print("[!!!] Ensure onnxruntime-gpu is installed and CUDA/cuDNN are in your PATH.\n")
        print(f"[*] Active Providers: {active_providers}")
        for inp in session.get_inputs():
            print(f"[*] Model Input: '{inp.name}' | Shape: {inp.shape} | Type: {inp.type}")
        for outp in session.get_outputs():
            print(f"[*] Model Output: '{outp.name}' | Shape: {outp.shape} | Type: {outp.type}")
        if 'CUDAExecutionProvider' not in session.get_providers():
            print("[!] Warning: CUDAExecutionProvider not active. Running on CPU.")

    mean, std = get_mean_std(args.dataset)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    gt_map = load_gt_data(args.gt)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) & ~1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) & ~1
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"[!] Error: Could not open VideoWriter for {args.output}")
        return
    print(f"[*] VideoWriter initialized: {width}x{height} @ {fps} FPS")

    tracker = SortPointTracker()
    frame_idx = 0
    times = []
    stats_raw = {'tp': 0, 'fp': 0, 'fn': 0, 'error': 0.0, 'matches': 0, 'frame_data': []}
    count_errors = {'mae_sum': 0.0, 'mse_sum': 0.0, 'n': 0}

    print(f"[*] Starting inference (Precision: {args.precision})...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        t0 = time.time()
        # Preprocess
        input_tensor = preprocess(frame, mean, std, args.precision)
        
        # Inference with tiling
        binar_map, _, _ = run_tiled_inference(
            session, 
            input_tensor, 
            batch_size=args.batch_size,
            verbose=(args.verbose and frame_idx == 0)
        )
        
        # Post-process
        pred_data, boxes = get_boxInfo_from_Binar_map(binar_map)
        t1 = time.time()
        dt = t1 - t0
        times.append(dt)

        # Tracking
        detections = np.array(pred_data['points']) if pred_data['num'] > 0 else np.empty((0, 2))
        tracked = tracker.update(detections)

        # Visualization
        if tracked.size > 0:
            for row in tracked:
                x, y, tid = int(row[0]), int(row[1]), int(row[2])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(frame, str(tid), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(frame, f'Count: {pred_data["num"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)

        # Metrics
        if args.gt:
            gt = gt_map.get(frame_idx, gt_map.get(str(frame_idx), None))
            if gt:
                gt_count = float(gt['num'])
                pred_count = float(pred_data['num'])
                count_errors['mae_sum'] += abs(gt_count - pred_count)
                count_errors['mse_sum'] += (gt_count - pred_count)**2
                count_errors['n'] += 1

                preds = np.array([row[:2] for row in tracked]) if tracked.size > 0 else detections
                gts = gt['points']
                
                m = {'tp': 0, 'fp': 0, 'fn': 0, 'error': 0.0, 'matches': 0}
                if preds.size > 0 and gts.size > 0:
                    d_matrix = scipy.spatial.distance.cdist(preds, gts, metric='euclidean')
                    r_ind, c_ind = scipy.optimize.linear_sum_assignment(d_matrix)
                    for r, c in zip(r_ind, c_ind):
                        if d_matrix[r, c] <= args.match_dist:
                            m['tp'] += 1
                            m['error'] += d_matrix[r, c]
                    m['fp'] = len(preds) - m['tp']
                    m['fn'] = len(gts) - m['tp']
                    m['matches'] = m['tp']
                elif gts.size > 0:
                    m['fn'] = len(gts)
                elif preds.size > 0:
                    m['fp'] = len(preds)

                stats_raw['tp'] += m['tp']
                stats_raw['fp'] += m['fp']
                stats_raw['fn'] += m['fn']
                stats_raw['error'] += m['error']
                stats_raw['matches'] += m['matches']
                
                p = m['tp'] / (m['tp'] + m['fp'] + 1e-9)
                r = m['tp'] / (m['tp'] + m['fn'] + 1e-9)
                f1 = 2*p*r / (p+r+1e-9)
                stats_raw['frame_data'].append({'p': p, 'r': r, 'f1': f1, 'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'], 'gt': gt_count, 'pred': pred_count})

        if args.verbose:
            print(f"Frame {frame_idx} | Count: {pred_data['num']} | Time: {dt:.4f}s")
        frame_idx += 1

    cap.release()
    out.release()
    
    # Final Reporting
    if len(times) > 0:
        avg_t = np.mean(times[1:]) if len(times) > 1 else times[0]
        print(f"\n[+] Finished. Avg Time: {avg_t:.4f}s ({1.0/avg_t:.2f} FPS)")
        
        if count_errors['n'] > 0:
            mae = count_errors['mae_sum'] / count_errors['n']
            mse = np.sqrt(count_errors['mse_sum'] / count_errors['n'])
            print(f"[+] Count MAE: {mae:.2f}, MSE: {mse:.2f}")
            
            tot_tp, tot_fp, tot_fn = stats_raw['tp'], stats_raw['fp'], stats_raw['fn']
            mic_p = tot_tp / (tot_tp + tot_fp + 1e-9)
            mic_r = tot_tp / (tot_tp + tot_fn + 1e-9)
            mic_f1 = 2*mic_p*mic_r / (mic_p + mic_r + 1e-9)
            print(f"[+] Detection Micro F1: {mic_f1:.4f} (Prec: {mic_p:.4f}, Rec: {mic_r:.4f})")
            
            if args.out_dir:
                save_reports(args, stats_raw, avg_t, mic_p, mic_r, mic_f1)

def save_reports(args, stats, avg_t, mic_p, mic_r, mic_f1):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    
    # JSON
    report = {
        'precision_mode': args.precision,
        'avg_fps': 1.0/avg_t,
        'micro_metrics': {'f1': mic_f1, 'precision': mic_p, 'recall': mic_r}
    }
    with open(os.path.join(out_dir, f"{base}_onnx_metrics.json"), 'w') as f:
        json.dump(report, f, indent=4)
        
    # CSV
    with open(os.path.join(out_dir, f"{base}_onnx_frames.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'gt', 'pred', 'tp', 'fp', 'fn', 'f1'])
        for i, d in enumerate(stats['frame_data']):
            writer.writerow([i, d['gt'], d['pred'], d['tp'], d['fp'], d['fn'], d['f1']])

if __name__ == '__main__':
    main()