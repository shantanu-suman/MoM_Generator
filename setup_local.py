#!/usr/bin/env python3
"""
Quick setup script for Growth Talk Assistant on local PC
This script will help you set up the environment and verify installation
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and report status"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8 or higher")
        return False

def install_packages():
    """Install required packages"""
    packages = [
        "streamlit>=1.28.0",
        "transformers>=4.30.0", 
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0"
    ]
    
    print(f"\n📦 Installing {len(packages)} required packages...")
    for package in packages:
        print(f"   Installing {package.split('>=')[0]}...")
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            return False
    
    return True

def create_streamlit_config():
    """Create Streamlit configuration"""
    print("\n⚙️ Creating Streamlit configuration...")
    
    # Create .streamlit directory
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create config.toml
    config_content = """[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
base = "light"
"""
    
    config_file = streamlit_dir / "config.toml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ Streamlit configuration created")
    return True

def verify_installation():
    """Verify that key packages are installed"""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        ("streamlit", "Streamlit web framework"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch"),
        ("pandas", "Pandas data analysis"),
        ("plotly", "Plotly visualization"),
        ("sklearn", "Scikit-learn machine learning")
    ]
    
    all_good = True
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description} - OK")
        except ImportError:
            print(f"❌ {description} - MISSING")
            all_good = False
    
    return all_good

def create_sample_vtt():
    """Create a sample VTT file for testing"""
    print("\n📝 Creating sample VTT file for testing...")
    
    sample_content = """WEBVTT

00:00:01.000 --> 00:00:05.000
Manager: Good morning! How are you feeling about your current projects?

00:00:06.000 --> 00:00:12.000
Employee: I'm doing well, though I've been struggling with the deadline on the analytics project.

00:00:13.000 --> 00:00:18.000
Manager: I understand. What specific challenges are you facing?

00:00:19.000 --> 00:00:25.000
Employee: The data integration is more complex than expected. I might need an extra week.

00:00:26.000 --> 00:00:32.000
Manager: That's perfectly reasonable. Let's discuss how we can support you better.

00:00:33.000 --> 00:00:38.000
Employee: Thank you for understanding. I really appreciate your support.

00:00:39.000 --> 00:00:44.000
Manager: Of course! Let's set up regular check-ins to ensure you have what you need.
"""
    
    with open("sample_meeting.vtt", 'w') as f:
        f.write(sample_content)
    
    print("✅ Sample VTT file created: sample_meeting.vtt")
    return True

def main():
    """Main setup process"""
    print("=" * 60)
    print("🚀 GROWTH TALK ASSISTANT - LOCAL SETUP")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Please install Python 3.8 or higher")
        return False
    
    # Step 2: Install packages
    if not install_packages():
        print("\n❌ Setup failed: Package installation error")
        return False
    
    # Step 3: Create Streamlit config
    if not create_streamlit_config():
        print("\n❌ Setup failed: Configuration error")
        return False
    
    # Step 4: Verify installation
    if not verify_installation():
        print("\n❌ Setup failed: Installation verification error")
        return False
    
    # Step 5: Create sample file
    create_sample_vtt()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Train the models:")
    print("   python train_models.py")
    print("\n2. Start the application:")
    print("   streamlit run app.py")
    print("\n3. Open your browser to:")
    print("   http://localhost:8501")
    print("\n4. Upload sample_meeting.vtt to test the system")
    
    print("\n💡 TIPS:")
    print("• Training takes 30-60 minutes")
    print("• Use GPU if available for faster training")
    print("• Check LOCAL_SETUP.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)