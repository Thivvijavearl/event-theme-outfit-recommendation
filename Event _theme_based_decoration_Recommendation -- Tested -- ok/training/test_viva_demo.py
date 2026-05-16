#!/usr/bin/env python
"""
StyleMate Color Classifier - VIVA Test Script
Test the color prediction system with multiple images
"""

import requests
import os
import sys
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# API endpoint
API_URL = "http://localhost:5002/predict"

# Test images
TEST_IMAGES = [
    (os.path.join(BASE_DIR, 'data', 'raw', 'Blue', 'Blue00001.jpg'), 'Blue'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Yellow', 'Yellow00001.jpg'), 'Yellow'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Red', 'Red00001.jpg'), 'Red'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Purble', 'Purble00006.jpg'), 'Purble'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Green', 'Green00001.jpg'), 'Green'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Black', 'Black00001.jpg'), 'Black'),
    (os.path.join(BASE_DIR, 'data', 'raw', 'Red-Yellow', 'Red-Yellow00001.jpg'), 'Red-Yellow'),
]

def print_header():
    """Print test header"""
    print("\n" + "="*70)
    print("  StyleMate AI Fashion Assistant - Color Classification Test")
    print("  VIVA Demo - Histogram-Based SVM Classifier")
    print("="*70)
    print(f"  Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def print_results_header():
    """Print results header"""
    print(f"{'Result':<8} {'Expected':<12} {'Predicted':<12} {'Confidence':<12} {'Status':<10}")
    print("-" * 70)

def test_color(image_path, expected_color):
    """Test prediction for a single image"""
    if not os.path.exists(image_path):
        print(f"{'✗':<8} {expected_color:<12} {'N/A':<12} {'0.0%':<12} File Not Found")
        return False

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(API_URL, files=files, timeout=10)
            
            if response.status_code != 200:
                print(f"{'✗':<8} {expected_color:<12} {'ERROR':<12} {'0.0%':<12} API Error")
                return False
            
            result = response.json()
            predicted = result.get('predicted_species', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            # Check if prediction is correct
            is_correct = predicted == expected_color
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            # Format output
            confidence_str = f"{confidence:.1f}%"
            result_str = "✓" if is_correct else "✗"
            
            print(f"{result_str:<8} {expected_color:<12} {predicted:<12} {confidence_str:<12} {status:<10}")
            
            return is_correct
            
    except requests.exceptions.ConnectionError:
        print(f"{'✗':<8} {expected_color:<12} {'ERROR':<12} {'0.0%':<12} Connection Failed")
        return False
    except Exception as e:
        print(f"{'✗':<8} {expected_color:<12} {'ERROR':<12} {'0.0%':<12} {str(e)[:20]}")
        return False

def test_api_health():
    """Test if API is healthy"""
    print("Checking API Status...\n")
    try:
        response = requests.get("http://localhost:5002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: HEALTHY")
            print(f"  - Model Loaded: {data.get('model_loaded', False)}")
            print(f"  - Label Encoder Loaded: {data.get('label_encoder_loaded', False)}")
            print(f"  - Endpoint Type: {data.get('endpoint_type', 'Unknown')}\n")
            return True
        else:
            print(f"✗ API Status: UNHEALTHY (Status Code: {response.status_code})\n")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ API Status: UNREACHABLE - Is the backend running?\n")
        print("  Start backend with: python MobileApi_Color.py\n")
        return False
    except Exception as e:
        print(f"✗ API Status: ERROR - {e}\n")
        return False

def main():
    """Main test function"""
    print_header()
    
    # Check API health
    if not test_api_health():
        print("="*70)
        print("ERROR: Backend API is not running!")
        print("Please start the backend with: python MobileApi_Color.py")
        print("="*70 + "\n")
        sys.exit(1)
    
    # Run tests
    print("Running Color Classification Tests...\n")
    print_results_header()
    
    passed = 0
    failed = 0
    
    for image_path, expected_color in TEST_IMAGES:
        if test_color(image_path, expected_color):
            passed += 1
        else:
            failed += 1
    
    # Print summary
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    
    print("-" * 70)
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    print(f"  Total Tests:     {total}")
    print(f"  Passed:          {passed} ✓")
    print(f"  Failed:          {failed} ✗")
    print(f"  Accuracy:        {accuracy:.1f}%")
    
    if accuracy == 100:
        print("\n  Status: ✓ ALL TESTS PASSED - READY FOR VIVA!")
    elif accuracy >= 85:
        print(f"\n  Status: ✓ GOOD - {accuracy:.1f}% Accuracy")
    else:
        print(f"\n  Status: ✗ NEEDS IMPROVEMENT - {accuracy:.1f}% Accuracy")
    
    print("="*70)
    print(f"  Test End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Return exit code based on results
    return 0 if accuracy >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())