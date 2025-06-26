#!/usr/bin/env python3
"""
Environment setup and check script
Check Python environment and dependencies (using Pipenv)
"""

import sys
import subprocess
import importlib
import os
import shutil
from pathlib import Path

def check_pipenv():
    """Check if Pipenv is installed"""
    print("🔧 Checking Pipenv...")
    try:
        result = subprocess.run(['pipenv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Pipenv is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pipenv is not installed")
            return False
    except FileNotFoundError:
        print("❌ Pipenv is not installed")
        return False

def install_pipenv():
    """Install Pipenv"""
    print("Installing Pipenv...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pipenv"])
        print("✅ Pipenv installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Pipenv: {e}")
        return False

def check_virtual_env():
    """Check if running in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    pipenv_active = os.environ.get('PIPENV_ACTIVE') == '1'
    
    if pipenv_active:
        print("✅ Currently in Pipenv virtual environment")
        return True
    elif in_venv:
        print("ℹ️  Currently in another virtual environment")
        return True
    else:
        print("⚠️  Not in a virtual environment")
        return False

def install_dependencies():
    """Install dependencies using Pipenv"""
    print("📦 Installing dependencies using Pipenv...")
    try:
        # Check if Pipfile exists
        if not Path("Pipfile").exists():
            print("❌ Pipfile not found")
            return False
        
        # Install dependencies
        result = subprocess.run(['pipenv', 'install'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Warning: Python 3.9 or higher is recommended")
        return False
    else:
        print("✅ Python version meets requirements")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "Unknown"
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def install_package(package_name):
    """Install a package"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_directories():
    """Check required directories"""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        "dataset",
        "dataset/train/cats",
        "dataset/train/dogs", 
        "dataset/validation/cats",
        "dataset/validation/dogs",
        "dataset/test/cats",
        "dataset/test/dogs",
        "models",
        "web_app",
        "web_app/templates",
        "web_app/uploads"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (does not exist)")
            all_exist = False
    
    return all_exist

def create_directories():
    """Create missing directories"""
    print("\n🔧 Creating missing directories...")
    
    required_dirs = [
        "dataset/train/cats",
        "dataset/train/dogs", 
        "dataset/validation/cats",
        "dataset/validation/dogs",
        "dataset/test/cats",
        "dataset/test/dogs",
        "models",
        "web_app/templates",
        "web_app/uploads"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")

def check_gpu():
    """Check GPU support"""
    print("\n🖥️  Checking GPU support...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU devices:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  No GPU detected, using CPU for computation")
        return True
    except ImportError:
        print("❌ TensorFlow is not installed, unable to check GPU")
        return False

def main():
    """Main function"""
    print("🚀 Cat and Dog Image Classification System - Environment Check (Pipenv)")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check Pipenv
    pipenv_installed = check_pipenv()
    if not pipenv_installed:
        install_choice = input("Do you want to install Pipenv? (y/n): ").lower().strip()
        if install_choice in ['y', 'yes']:
            pipenv_installed = install_pipenv()
        else:
            print("❌ Pipenv is required to proceed")
            return
    
    # Check virtual environment
    check_virtual_env()
    
    # Check if dependencies need to be installed
    if pipenv_installed:
        pipfile_lock_exists = Path("Pipfile.lock").exists()
        if not pipfile_lock_exists:
            print("\n📦 Detected new Pipfile, dependencies need to be installed")
            install_choice = input("Do you want to install project dependencies? (y/n): ").lower().strip()
            if install_choice in ['y', 'yes']:
                deps_installed = install_dependencies()
            else:
                print("⚠️  Please manually run 'pipenv install' to install dependencies")
                deps_installed = False
        else:
            print("✅ Pipfile.lock exists, dependencies are installed")
            deps_installed = True
    
    # Check directory structure
    dirs_ok = check_directories()
    if not dirs_ok:
        create_directories()
    
    # Final report
    print("\n" + "=" * 50)
    print("📋 Environment check completed")
    
    if python_ok and pipenv_installed:
        print("✅ Basic environment setup completed!")
        print("\n🎯 Next steps:")
        print("1. Run 'pipenv shell' to enter the virtual environment")
        print("2. Run 'pipenv run prepare-data' to prepare the dataset")
        print("3. Run 'pipenv run train' to train the model")
        print("4. Run 'pipenv run predict <image_path>' to test predictions")
        print("5. Run 'pipenv run web' to start the web application")
        
        print("\n🛠️  Common Pipenv commands:")
        print("- pipenv shell          # Enter the virtual environment")
        print("- pipenv install        # Install all dependencies")
        print("- pipenv install <pkg>  # Install a new package")
        print("- pipenv install --dev  # Install development dependencies")
        print("- pipenv run <command>  # Run a command in the virtual environment")
        print("- pipenv graph          # Show dependency graph")
        print("- pipenv check          # Check for security vulnerabilities")
    else:
        print("⚠️  Environment setup incomplete, please resolve the issues above and rerun")
    
    print("\n💡 Tips:")
    print("- Use Pipenv to manage virtual environments and dependencies")
    print("- All Python commands should be run in the 'pipenv shell' environment")
    print("- Or use 'pipenv run <command>' to execute commands")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Please check the error message and rerun")
