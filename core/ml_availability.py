"""
ML package availability checker
Gracefully handles missing ML dependencies during minimal deployment
"""

def check_ml_packages():
    """Check which ML packages are available"""
    available = {
        'numpy': False,
        'pandas': False,
        'sklearn': False,
        'tensorflow': False,
        'opencv': False,
    }
    
    try:
        import numpy
        available['numpy'] = True
    except ImportError:
        pass
    
    try:
        import pandas
        available['pandas'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        available['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        available['tensorflow'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        available['opencv'] = True
    except ImportError:
        pass
    
    return available

def ml_available():
    """Check if all required ML packages are available"""
    packages = check_ml_packages()
    return all(packages.values())

def get_ml_status():
    """Get detailed ML package status"""
    packages = check_ml_packages()
    missing = [pkg for pkg, available in packages.items() if not available]
    
    return {
        'available': packages,
        'all_available': len(missing) == 0,
        'missing': missing,
        'message': 'All ML packages available' if len(missing) == 0 else f'Missing: {", ".join(missing)}'
    }