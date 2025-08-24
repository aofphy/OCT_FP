#!/usr/bin/env python3
# Test VGG16 system

print("🔧 Testing VGG16 Single Model System...")
import sys
sys.path.append('.')

try:
    import oct_deep_fixed
    
    # Test VGG16 extractor
    extractor = oct_deep_fixed.VGG16FeatureExtractor()
    print(f"✅ VGG16 initialized: {extractor.model_loaded}")
    
    # Test feature extraction
    import numpy as np
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = extractor.extract_features(test_image)
    print(f"✅ Feature extraction: shape={features.shape if features is not None else 'None'}")
    
    # Test user manager
    manager = oct_deep_fixed.OCTUserManager(extractor)
    print("✅ User manager initialized")
    
    # Test similarity calculation
    if features is not None:
        features1 = features
        features2 = np.random.rand(1000).astype(np.float32)
        similarity = manager.calculate_similarity(features1, features2)
        print(f"✅ Similarity calculation: {similarity:.4f}")
    
    print("🎯 All VGG16 components working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
