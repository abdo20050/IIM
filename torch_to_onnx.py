import os
import torch
import argparse
from collections import OrderedDict
from model.locator import Crowd_locator

def main():
    parser = argparse.ArgumentParser(description="Convert IIM PyTorch models to ONNX format")
    parser.add_argument('--model', '-m', required=True, help='Path to the pretrained .pth model')
    parser.add_argument('--net', default='HR_Net', choices=['HR_Net', 'VGG16_FPN'], help='Network architecture name')
    parser.add_argument('--output', '-o', help='Path to save the ONNX model (default: same as model with .onnx)')
    parser.add_argument('--gpu', default='0', help='GPU ID for model initialization')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.model)[0] + ".onnx"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize the model
    # We set pretrained=False because we are loading a specific checkpoint
    print(f"[*] Initializing model: {args.net}")
    net = Crowd_locator(args.net, args.gpu, pretrained=False)
    
    # 2. Load the weights (Logic mirrored from pred.py)
    print(f"[*] Loading weights from: {args.model}")
    if not os.path.exists(args.model):
        print(f"[!] Error: Model file {args.model} not found.")
        return

    state = torch.load(args.model, map_location='cpu')
    
    if isinstance(state, dict) and ('net' in state or 'state_dict' in state):
        sd = state.get('net', state.get('state_dict', state))
    else:
        sd = state

    # Strip 'module.' prefix if it exists (common in DataParallel checkpoints)
    new_state = OrderedDict()
    for k, v in sd.items():
        name = k.replace('module.', '')
        new_state[name] = v
        
    try:
        net.load_state_dict(new_state)
    except Exception:
        try:
            net.load_state_dict(sd)
        except Exception as e:
            print(f"[!] Model loading failed: {e}")
            return

    net.to(device)
    net.eval()

    # 3. Prepare dummy input
    # Standard slice size used in inference is 512x1024
    dummy_input = torch.randn(1, 3, 512, 1024).to(device)
    
    # 4. Export to ONNX
    # The forward call in pred.py is: net(img, mask_gt=None, mode='val')
    # We pass these as a tuple to the export function.
    print(f"[*] Exporting to {args.output} (Opset {args.opset})...")
    
    input_names = ["input"]
    output_names = ["pred_threshold", "pred_map", "extra_info"]
    
    try:
        torch.onnx.export(
            net,
            (dummy_input, None, 'val'),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch', 2: 'height', 3: 'width'},
                'pred_threshold': {0: 'batch', 2: 'height', 3: 'width'},
                'pred_map': {0: 'batch', 2: 'height', 3: 'width'}
            }
        )
        print("[+] Conversion successful!")
    except Exception as e:
        print(f"[!] Export failed: {e}")

if __name__ == '__main__':
    main()