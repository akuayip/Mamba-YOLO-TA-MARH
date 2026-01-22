#!/usr/bin/env python3
"""
Check YOLO Dataset Format - Detection vs Segmentation
Identifies which label files contain segmentation annotations
"""

import os
from pathlib import Path
from collections import defaultdict


def check_label_format(label_path):
    """
    Check if label file contains detection or segmentation format
    
    Detection format: class_id x_center y_center width height (5 values)
    Segmentation format: class_id x1 y1 x2 y2 ... xn yn (5+ values)
    
    Returns:
        tuple: (format_type, line_count, details)
        format_type: 'detection', 'segmentation', or 'mixed'
    """
    detection_lines = 0
    segmentation_lines = 0
    details = []
    
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                num_values = len(parts)
                
                if num_values == 5:
                    # Detection format: class x_center y_center width height
                    detection_lines += 1
                elif num_values > 5:
                    # Segmentation format: class x1 y1 x2 y2 ... (polygon points)
                    segmentation_lines += 1
                    details.append(f"  Line {line_num}: {num_values} values (segmentation)")
                else:
                    # Invalid format
                    details.append(f"  Line {line_num}: {num_values} values (INVALID)")
        
        if segmentation_lines > 0 and detection_lines > 0:
            return 'mixed', detection_lines + segmentation_lines, details
        elif segmentation_lines > 0:
            return 'segmentation', segmentation_lines, details
        else:
            return 'detection', detection_lines, details
            
    except Exception as e:
        return 'error', 0, [f"Error reading file: {e}"]


def scan_dataset(dataset_root, splits=['train', 'valid', 'test']):
    """
    Scan entire dataset and categorize label files
    
    Args:
        dataset_root: Root directory of dataset
        splits: List of splits to check (train, valid, test)
    
    Returns:
        dict: Statistics and problematic files
    """
    results = {
        'detection': [],
        'segmentation': [],
        'mixed': [],
        'error': [],
        'stats': defaultdict(lambda: {'detection': 0, 'segmentation': 0, 'mixed': 0, 'error': 0})
    }
    
    for split in splits:
        labels_dir = Path(dataset_root) / split / 'labels'
        
        if not labels_dir.exists():
            print(f"⚠ Warning: {labels_dir} does not exist. Skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Scanning: {split}/labels/")
        print(f"{'='*60}")
        
        label_files = sorted(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            format_type, line_count, details = check_label_format(label_file)
            
            # Store results
            results[format_type].append({
                'file': str(label_file),
                'split': split,
                'lines': line_count,
                'details': details
            })
            results['stats'][split][format_type] += 1
        
        print(f"✓ Scanned {len(label_files)} files in {split}")
    
    return results


def print_report(results):
    """Print detailed report"""
    
    print(f"\n{'='*60}")
    print("DATASET FORMAT CHECK REPORT")
    print(f"{'='*60}\n")
    
    # Summary statistics
    print("📊 Summary by Split:")
    print("-" * 60)
    for split, stats in results['stats'].items():
        print(f"\n{split.upper()}:")
        print(f"  ✓ Detection only:    {stats['detection']:>5} files")
        print(f"  ⚠ Segmentation only: {stats['segmentation']:>5} files")
        print(f"  ❌ Mixed format:      {stats['mixed']:>5} files")
        print(f"  ⚠ Errors:            {stats['error']:>5} files")
    
    # Overall statistics
    total_detection = len(results['detection'])
    total_segmentation = len(results['segmentation'])
    total_mixed = len(results['mixed'])
    total_error = len(results['error'])
    total_files = total_detection + total_segmentation + total_mixed + total_error
    
    print(f"\n{'='*60}")
    print("📈 Overall Statistics:")
    print("-" * 60)
    print(f"Total files:          {total_files}")
    print(f"  Detection format:   {total_detection} ({total_detection/total_files*100:.1f}%)")
    print(f"  Segmentation format: {total_segmentation} ({total_segmentation/total_files*100:.1f}%)")
    print(f"  Mixed format:       {total_mixed} ({total_mixed/total_files*100:.1f}%)")
    print(f"  Errors:             {total_error} ({total_error/total_files*100:.1f}%)")
    
    # Problem files
    if total_segmentation > 0 or total_mixed > 0:
        print(f"\n{'='*60}")
        print("⚠️  PROBLEMATIC FILES (Segmentation/Mixed):")
        print(f"{'='*60}")
        
        # Segmentation files
        if total_segmentation > 0:
            print(f"\n🔷 Segmentation-only files ({total_segmentation}):")
            for item in results['segmentation'][:10]:  # Show first 10
                print(f"  - {Path(item['file']).name} ({item['split']}) - {item['lines']} annotations")
            if total_segmentation > 10:
                print(f"  ... and {total_segmentation - 10} more")
        
        # Mixed files
        if total_mixed > 0:
            print(f"\n❌ Mixed format files ({total_mixed}):")
            for item in results['mixed']:
                print(f"\n  File: {Path(item['file']).name} ({item['split']})")
                for detail in item['details']:
                    print(f"    {detail}")
    
    # Errors
    if total_error > 0:
        print(f"\n{'='*60}")
        print(f"⚠️  FILES WITH ERRORS ({total_error}):")
        print(f"{'='*60}")
        for item in results['error']:
            print(f"  - {Path(item['file']).name}: {item['details'][0]}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    if total_segmentation > 0 or total_mixed > 0:
        print("\n⚠️  Your dataset contains SEGMENTATION annotations!")
        print("\nOptions to fix:")
        print("  1. Convert segmentation to detection (bbox only)")
        print("  2. Remove segmentation files")
        print("  3. Use separate segmentation model (yolo-seg)")
        print(f"\nGenerate fix script: python check_dataset_format.py --fix")
    else:
        print("\n✅ Dataset is clean! All files use detection format.")
        print("   Ready for object detection training.")


def save_report(results, output_file='dataset_format_report.txt'):
    """Save report to file"""
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATASET FORMAT CHECK - DETAILED REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Segmentation files
        if results['segmentation']:
            f.write(f"SEGMENTATION FILES ({len(results['segmentation'])}):\n")
            f.write("-"*60 + "\n")
            for item in results['segmentation']:
                f.write(f"{item['file']}\n")
            f.write("\n")
        
        # Mixed files
        if results['mixed']:
            f.write(f"MIXED FORMAT FILES ({len(results['mixed'])}):\n")
            f.write("-"*60 + "\n")
            for item in results['mixed']:
                f.write(f"{item['file']}\n")
                for detail in item['details']:
                    f.write(f"  {detail}\n")
            f.write("\n")
        
        # Error files
        if results['error']:
            f.write(f"ERROR FILES ({len(results['error'])}):\n")
            f.write("-"*60 + "\n")
            for item in results['error']:
                f.write(f"{item['file']}: {item['details'][0]}\n")
            f.write("\n")
    
    print(f"\n✓ Detailed report saved to: {output_file}")


def generate_fix_script(results, output_file='fix_dataset.sh'):
    """Generate bash script to remove problematic files"""
    problematic = results['segmentation'] + results['mixed']
    
    if not problematic:
        print("\n✅ No problematic files to fix!")
        return
    
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to remove segmentation/mixed format files\n")
        f.write(f"# Total files to remove: {len(problematic)}\n\n")
        f.write("echo 'Removing problematic label files...'\n\n")
        
        for item in problematic:
            label_path = item['file']
            # Also remove corresponding image
            image_path = label_path.replace('/labels/', '/images/').replace('.txt', '.jpg')
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpg', '.png')
            
            f.write(f"# {Path(label_path).name}\n")
            f.write(f"rm -f '{label_path}'\n")
            if os.path.exists(image_path):
                f.write(f"rm -f '{image_path}'\n")
            f.write("\n")
        
        f.write(f"echo 'Removed {len(problematic)} label files and their images'\n")
        f.write("echo 'Done!'\n")
    
    # Make executable
    os.chmod(output_file, 0o755)
    
    print(f"\n✓ Fix script generated: {output_file}")
    print(f"  To remove {len(problematic)} problematic files, run:")
    print(f"  bash {output_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check YOLO dataset format')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'], help='Splits to check')
    parser.add_argument('--fix', action='store_true', help='Generate fix script')
    parser.add_argument('--report', type=str, default='dataset_format_report.txt', help='Report output file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("YOLO Dataset Format Checker")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"{'='*60}")
    
    # Scan dataset
    results = scan_dataset(args.dataset, args.splits)
    
    # Print report
    print_report(results)
    
    # Save detailed report
    save_report(results, args.report)
    
    # Generate fix script if requested
    if args.fix:
        generate_fix_script(results)


if __name__ == '__main__':
    main()
