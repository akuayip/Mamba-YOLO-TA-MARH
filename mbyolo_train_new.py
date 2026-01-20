#!/usr/bin/env python3
"""
Mamba-YOLO Training Script with Transfer Learning Support
Supports: train, val, test, and sequential execution (all)
"""

from ultralytics import YOLO
import argparse
import os
import time
import torch

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Mamba-YOLO Training Script')
    
    # Dataset & Model
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='dataset config path')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model config path')
    parser.add_argument('--weights', type=str, default=None, help='trained model weights for val/test')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained weights (e.g., yolov8n.pt)')
    
    # Training Parameters
    parser.add_argument('--task', nargs='+', default=['train'], help='task: train, val, test, or all')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    
    # Hyperparameters
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--optimizer', default='SGD', help='optimizer: SGD, Adam, AdamW')
    
    # Advanced Options
    parser.add_argument('--freeze-backbone', action='store_true', help='freeze backbone layers')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    
    # System
    parser.add_argument('--device', default='0', help='cuda device: 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--project', default='output_dir/head_detection', help='project folder')
    parser.add_argument('--name', default='mambayolo_head', help='experiment name')
    
    return parser.parse_args()


def transfer_weights(model, pretrained_path, freeze_backbone=False):
    """Transfer compatible weights from pretrained model"""
    print(f"\n{'='*60}")
    print(f"Transfer Learning: {pretrained_path}")
    print(f"Freeze Backbone: {freeze_backbone}")
    print(f"{'='*60}")
    
    try:
        # Load pretrained weights
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        source_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
        target_dict = model.model.state_dict()
        
        # Transfer compatible weights
        transferred, skipped = 0, 0
        for name, param in target_dict.items():
            if name in source_dict and param.shape == source_dict[name].shape:
                target_dict[name] = source_dict[name]
                transferred += 1
            else:
                skipped += 1
        
        model.model.load_state_dict(target_dict, strict=False)
        print(f"✓ Transferred: {transferred} layers | ✗ Skipped: {skipped} layers")
        
        # Freeze backbone if requested
        if freeze_backbone:
            frozen = 0
            for name, param in model.model.named_parameters():
                if 'model.22' not in name:  # Don't freeze detection head
                    param.requires_grad = False
                    frozen += 1
            print(f"✓ Froze {frozen} parameters (backbone only)")
        
        print(f"{'='*60}\n")
        return model
        
    except Exception as e:
        print(f"⚠ Warning: Transfer failed ({e}). Using random initialization.\n")
        return model


def calculate_stats(model, imgsz=640, device='0'):
    """Calculate model statistics: GFLOPs, Parameters, Latency"""
    try:
        from thop import profile
        
        # Setup device
        dev = torch.device('cpu' if device == 'cpu' else f'cuda:{device}')
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(dev)
        net = model.model.to(dev).eval()
        
        # Calculate FLOPs and Params
        flops, params = profile(net, inputs=(dummy_input,), verbose=False)
        gflops, params_m = flops / 1e9, params / 1e6
        
        # Calculate Latency (100 runs)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = net(dummy_input)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                _ = net(dummy_input)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        return gflops, params_m, avg_latency
        
    except Exception as e:
        print(f"⚠ Could not calculate stats: {e}")
        return None, None, None


def run_train(opt):
    """Execute training task"""
    model_path = ROOT + opt.config
    
    # Training arguments
    args = {
        "data": ROOT + opt.data,
        "epochs": opt.epochs,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "lr0": opt.lr0,
        "lrf": opt.lrf,
        "momentum": opt.momentum,
        "weight_decay": opt.weight_decay,
        "optimizer": opt.optimizer,
        "amp": opt.amp,
        "resume": opt.resume,
        "device": opt.device,
        "workers": opt.workers,
        "project": ROOT + opt.project,
        "name": opt.name,
    }
    
    # Print config
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {opt.config}")
    print(f"  Pretrained: {opt.pretrained or 'None (from scratch)'}")
    print(f"  Epochs: {opt.epochs} | Batch: {opt.batch_size} | Image: {opt.imgsz}")
    print(f"  LR: {opt.lr0} → {opt.lr0 * opt.lrf} | Optimizer: {opt.optimizer}")
    print(f"  Device: {opt.device} | AMP: {opt.amp} | Freeze: {opt.freeze_backbone}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Apply transfer learning
    if opt.pretrained:
        model = transfer_weights(model, ROOT + opt.pretrained, opt.freeze_backbone)
    
    # Train
    model.train(**args)
    
    best_weights = f"{opt.project}/{opt.name}/weights/best.pt"
    print(f"\n✓ Training completed! Best model: {best_weights}\n")
    return best_weights


def run_val(opt, weights=None):
    """Execute validation task"""
    model_path = weights or (ROOT + opt.weights if opt.weights else ROOT + opt.config)
    
    args = {
        "data": ROOT + opt.data,
        "split": "val",
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "workers": opt.workers,
        "project": ROOT + opt.project,
        "name": opt.name,
    }
    
    print(f"\n{'='*60}")
    print(f"Validation: {model_path}")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    results = model.val(**args)
    
    # Calculate stats
    gflops, params_m, latency = calculate_stats(model, opt.imgsz, opt.device)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  mAP50: {results.box.map50:.4f} | mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f} | Recall: {results.box.mr:.4f}")
    if gflops:
        print(f"\nModel Statistics:")
        print(f"  Parameters: {params_m:.2f}M | GFLOPs: {gflops:.2f}")
        print(f"  Latency: {latency:.2f}ms | FPS: {1000/latency:.2f}")
    print(f"{'='*60}\n")


def run_test(opt, weights=None):
    """Execute test task"""
    model_path = weights or (ROOT + opt.weights if opt.weights else ROOT + opt.config)
    
    args = {
        "data": ROOT + opt.data,
        "split": "test",
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "workers": opt.workers,
        "project": ROOT + opt.project,
        "name": "test_results",
    }
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    results = model.val(**args)
    
    # Calculate stats
    gflops, params_m, latency = calculate_stats(model, opt.imgsz, opt.device)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  mAP50: {results.box.map50:.4f} | mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f} | Recall: {results.box.mr:.4f}")
    if gflops:
        print(f"\nModel Statistics:")
        print(f"  Parameters: {params_m:.2f}M | GFLOPs: {gflops:.2f}")
        print(f"  Latency: {latency:.2f}ms | FPS: {1000/latency:.2f}")
    print(f"{'='*60}\n")


def main():
    """Main execution"""
    opt = parse_opt()
    
    # Handle 'all' task
    tasks = opt.task
    if 'all' in tasks:
        tasks = ['train', 'val', 'test']
    
    print(f"\n{'='*60}")
    print(f"Mamba-YOLO Training Pipeline")
    print(f"Tasks: {' → '.join(tasks)}")
    print(f"{'='*60}")
    
    # Execute tasks sequentially
    trained_weights = None
    
    for task in tasks:
        if task == 'train':
            trained_weights = run_train(opt)
        elif task == 'val':
            run_val(opt, trained_weights)
        elif task == 'test':
            run_test(opt, trained_weights)
        else:
            print(f"⚠ Unknown task: {task}")
    
    print(f"\n{'='*60}")
    print(f"All tasks completed!")
    if trained_weights:
        print(f"Trained model: {trained_weights}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
