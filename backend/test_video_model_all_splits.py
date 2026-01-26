"""
Quick Video Model Test - Tests ALL splits (train/val/test) and classes (REAL/FAKE)
Provides detailed breakdown of detection accuracy across all folders
"""
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from models.video_detector import VideoDetector
from preprocessing.video_preprocessor import VideoPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoModelTester:
    """Test video model across all dataset splits"""
    
    def __init__(self, model_path, device='cpu'):
        """Initialize tester"""
        self.device = torch.device(device)
        self.model = VideoDetector(
            model_path=model_path,
            device=str(self.device)
        )
        self.preprocessor = VideoPreprocessor(device=str(self.device))
        
        logger.info(f"✓ Model loaded on {self.device}")
    
    def test_split(self, dataset_root: str, split: str = 'test'):
        """
        Test model on a specific split (train/validation/test)
        
        Args:
            dataset_root: Path to dataset root
            split: 'train', 'validation', or 'test'
        
        Returns:
            Results dictionary with metrics for each class
        """
        dataset_path = Path(dataset_root) / 'video' / 'test'
        
        if not dataset_path.exists():
            logger.error(f"❌ Split path not found: {dataset_path}")
            return None
        
        results = {
            'split': split,
            'REAL': {'correct': 0, 'total': 0, 'videos': []},
            'FAKE': {'correct': 0, 'total': 0, 'videos': []}
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {split.upper()} split")
        logger.info(f"{'='*70}")
        
        # Test REAL videos
        real_dir = dataset_path / 'REAL'
        if real_dir.exists():
            logger.info(f"\n📹 Testing REAL videos...")
            real_videos = list(real_dir.glob('*'))
            
            for video_path in tqdm(real_videos, desc=f"REAL {split}"):
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    try:
                        # Preprocess and predict
                        preprocess_result = self.preprocessor.preprocess_video(
                            str(video_path),
                            detect_faces=True
                        )
                        
                        if preprocess_result['status'] == 'success':
                            video_tensor = preprocess_result['video_tensor']
                            prediction = self.model.predict(video_tensor)
                            
                            # Check if correctly classified as REAL
                            is_correct = prediction['prediction'] == 'REAL'
                            
                            results['REAL']['total'] += 1
                            if is_correct:
                                results['REAL']['correct'] += 1
                            
                            results['REAL']['videos'].append({
                                'name': video_path.name,
                                'prediction': prediction['prediction'],
                                'confidence': prediction['confidence'],
                                'correct': is_correct
                            })
                    
                    except Exception as e:
                        logger.warning(f"Error processing {video_path.name}: {e}")
                        results['REAL']['total'] += 1
                        results['REAL']['videos'].append({
                            'name': video_path.name,
                            'error': str(e),
                            'correct': False
                        })
        
        # Test FAKE videos
        fake_dir = dataset_path / 'FAKE'
        if fake_dir.exists():
            logger.info(f"\n📹 Testing FAKE videos...")
            fake_videos = list(fake_dir.glob('*'))
            
            for video_path in tqdm(fake_videos, desc=f"FAKE {split}"):
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    try:
                        # Preprocess and predict
                        preprocess_result = self.preprocessor.preprocess_video(
                            str(video_path),
                            detect_faces=True
                        )
                        
                        if preprocess_result['status'] == 'success':
                            video_tensor = preprocess_result['video_tensor']
                            prediction = self.model.predict(video_tensor)
                            
                            # Check if correctly classified as FAKE
                            is_correct = prediction['prediction'] == 'FAKE'
                            
                            results['FAKE']['total'] += 1
                            if is_correct:
                                results['FAKE']['correct'] += 1
                            
                            results['FAKE']['videos'].append({
                                'name': video_path.name,
                                'prediction': prediction['prediction'],
                                'confidence': prediction['confidence'],
                                'correct': is_correct
                            })
                    
                    except Exception as e:
                        logger.warning(f"Error processing {video_path.name}: {e}")
                        results['FAKE']['total'] += 1
                        results['FAKE']['videos'].append({
                            'name': video_path.name,
                            'error': str(e),
                            'correct': False
                        })
        
        return results
    
    def print_results(self, results):
        """Pretty print results"""
        if not results:
            logger.error("No results to print")
            return
        
        split = results['split'].upper()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RESULTS FOR {split} SPLIT")
        logger.info(f"{'='*70}\n")
        
        # REAL results
        real_data = results['REAL']
        real_acc = (real_data['correct'] / real_data['total'] * 100) if real_data['total'] > 0 else 0
        
        logger.info(f"✅ REAL Videos:")
        logger.info(f"   Correctly Detected: {real_data['correct']}/{real_data['total']} ({real_acc:.2f}%)")
        logger.info(f"   Incorrectly Detected: {real_data['total'] - real_data['correct']}/{real_data['total']}")
        
        # Show some examples
        incorrect_real = [v for v in real_data['videos'] if not v.get('correct', False)]
        if incorrect_real:
            logger.info(f"\n   ⚠️  Misclassified REAL videos (showing first 3):")
            for video in incorrect_real[:3]:
                logger.info(f"      • {video['name']}: Predicted as {video.get('prediction', 'ERROR')} "
                          f"(confidence: {video.get('confidence', 0):.2f})")
        
        # FAKE results
        fake_data = results['FAKE']
        fake_acc = (fake_data['correct'] / fake_data['total'] * 100) if fake_data['total'] > 0 else 0
        
        logger.info(f"\n🔴 FAKE Videos:")
        logger.info(f"   Correctly Detected: {fake_data['correct']}/{fake_data['total']} ({fake_acc:.2f}%)")
        logger.info(f"   Incorrectly Detected: {fake_data['total'] - fake_data['correct']}/{fake_data['total']}")
        
        # Show some examples
        incorrect_fake = [v for v in fake_data['videos'] if not v.get('correct', False)]
        if incorrect_fake:
            logger.info(f"\n   ⚠️  Misclassified FAKE videos (showing first 3):")
            for video in incorrect_fake[:3]:
                logger.info(f"      • {video['name']}: Predicted as {video.get('prediction', 'ERROR')} "
                          f"(confidence: {video.get('confidence', 0):.2f})")
        
        # Overall accuracy
        total_correct = real_data['correct'] + fake_data['correct']
        total_videos = real_data['total'] + fake_data['total']
        overall_acc = (total_correct / total_videos * 100) if total_videos > 0 else 0
        
        logger.info(f"\n📈 OVERALL ACCURACY FOR {split}:")
        logger.info(f"   Total Videos: {total_videos}")
        logger.info(f"   Correct: {total_correct}/{total_videos} ({overall_acc:.2f}%)")
        logger.info(f"   Incorrect: {total_videos - total_correct}/{total_videos}")
        
        logger.info(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test video model on all splits')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset root path')
    parser.add_argument('--model', type=str, default='checkpoints/video/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("🎬 VIDEO MODEL COMPREHENSIVE TEST")
    logger.info("="*70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info("="*70)
    
    # Initialize tester
    tester = VideoModelTester(args.model, device=args.device)
    
    # Test all splits
    all_results = {}
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        results = tester.test_split(args.dataset, split)
        if results:
            all_results[split] = results
            tester.print_results(results)
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("📊 FINAL SUMMARY ACROSS ALL SPLITS")
    logger.info("="*70)
    
    summary_table = []
    for split in splits:
        if split in all_results:
            results = all_results[split]
            real_acc = (results['REAL']['correct'] / max(results['REAL']['total'], 1) * 100)
            fake_acc = (results['FAKE']['correct'] / max(results['FAKE']['total'], 1) * 100)
            total_acc = ((results['REAL']['correct'] + results['FAKE']['correct']) / 
                        max(results['REAL']['total'] + results['FAKE']['total'], 1) * 100)
            
            summary_table.append({
                'Split': split.upper(),
                'REAL Acc': f"{real_acc:.2f}%",
                'FAKE Acc': f"{fake_acc:.2f}%",
                'Overall Acc': f"{total_acc:.2f}%",
                'Total Videos': results['REAL']['total'] + results['FAKE']['total']
            })
    
    # Print table
    if summary_table:
        logger.info(f"\n{'Split':<15} {'REAL Acc':<15} {'FAKE Acc':<15} {'Overall Acc':<15} {'Total Videos':<15}")
        logger.info("-" * 75)
        for row in summary_table:
            logger.info(f"{row['Split']:<15} {row['REAL Acc']:<15} {row['FAKE Acc']:<15} "
                       f"{row['Overall Acc']:<15} {row['Total Videos']:<15}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Testing Complete!")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
