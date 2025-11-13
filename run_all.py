"""
Master script to run the entire sentiment classification pipeline
"""
import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(description, command):
    """Run a command and track execution time"""
    print(f"▶ {description}...")
    start_time = time.time()
    
    exit_code = os.system(command)
    
    elapsed = time.time() - start_time
    if exit_code == 0:
        print(f"✓ Completed in {elapsed:.1f}s\n")
        return True
    else:
        print(f"✗ Failed with exit code {exit_code}\n")
        return False

def setup_directories():
    """Create necessary directories"""
    print_header("SETTING UP DIRECTORIES")
    
    directories = ['data', 'results', 'results/plots']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created/verified: {dir_name}/")
    
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        'torch',
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies satisfied\n")
    return True

def run_full_pipeline(n_epochs=10):
    """
    Run the complete experimental pipeline
    
    Args:
        n_epochs: Number of training epochs per experiment
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print("  SENTIMENT CLASSIFICATION - FULL PIPELINE")
    print("  " + "="*68)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Epochs per experiment: {n_epochs}")
    print("="*70)
    
    # Step 0: Setup
    setup_directories()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install missing dependencies before continuing.")
        return False
    
    # Step 2: Print system info
    print_header("SYSTEM INFORMATION")
    from utils import print_system_info, save_system_info
    print_system_info()
    save_system_info()
    
    # Step 3: Preprocess data
    print_header("STEP 1: DATA PREPROCESSING")
    print("Preprocessing data for sequence lengths: 25, 50, 100")
    success = run_command(
        "Running preprocessing",
        f"{sys.executable} preprocess.py"
    )
    if not success:
        print("❌ Preprocessing failed. Exiting.")
        return False
    
    # Step 4: Run experiments
    print_header("STEP 2: RUNNING EXPERIMENTS")
    print(f"This will run 45+ experiments with {n_epochs} epochs each")
    print("Estimated time: 2-4 hours on CPU\n")
    
    response = input("Continue? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("Aborted by user.")
        return False
    
    success = run_command(
        f"Running all experiments ({n_epochs} epochs each)",
        f"{sys.executable} -c \"from run_experiments import run_all_experiments; run_all_experiments(n_epochs={n_epochs})\""
    )
    
    if not success:
        print("❌ Experiments failed. Check error messages above.")
        return False
    
    # Step 5: Generate visualizations
    print_header("STEP 3: GENERATING VISUALIZATIONS")
    success = run_command(
        "Creating plots and analysis",
        f"{sys.executable} evaluate.py"
    )
    
    if not success:
        print("⚠ Visualization generation failed, but experiments completed successfully.")
    
    # Summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print_header("PIPELINE COMPLETED")
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
    print(f"\nResults saved to:")
    print(f"  • results/metrics.csv         - Summary table")
    print(f"  • results/all_results.pkl     - Complete results")
    print(f"  • results/analysis.txt        - Text analysis")
    print(f"  • results/plots/              - Visualizations")
    print(f"  • results/system_info.txt     - Hardware specs")
    print("\n" + "="*70 + "\n")
    
    return True

def run_quick_test():
    """Run a quick test with minimal configurations"""
    print_header("QUICK TEST MODE")
    print("Running 3 configurations with 3 epochs each")
    print("Estimated time: 10-15 minutes\n")
    
    start_time = time.time()
    
    # Setup
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Preprocess (just one length)
    print_header("PREPROCESSING")
    from preprocess import preprocess_data
    preprocess_data(num_words=10000, max_length_list=[50])
    
    # Run quick test
    print_header("RUNNING QUICK TEST")
    from run_experiments import run_quick_test
    results = run_quick_test(n_epochs=3)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n✓ Quick test completed in {elapsed:.1f}s")
    print("\nTo run full experiments, use: python run_all.py full")
    
    return True

def print_usage():
    """Print usage instructions"""
    print("""
Usage:
    python run_all.py [mode] [options]

Modes:
    quick           Run quick test (3 configs, 3 epochs, ~15 min)
    full            Run full pipeline (45+ configs, 10 epochs, ~3 hours)
    full --epochs N Run full pipeline with N epochs

Examples:
    python run_all.py quick
    python run_all.py full
    python run_all.py full --epochs 5

Options:
    --epochs N      Number of training epochs (default: 10)
    --help, -h      Show this help message
""")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode in ['--help', '-h', 'help']:
        print_usage()
        sys.exit(0)
    
    elif mode == 'quick':
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    elif mode == 'full':
        n_epochs = 10
        if len(sys.argv) > 2 and sys.argv[2] == '--epochs':
            try:
                n_epochs = int(sys.argv[3])
            except (IndexError, ValueError):
                print("Error: --epochs requires an integer value")
                sys.exit(1)
        
        success = run_full_pipeline(n_epochs=n_epochs)
        sys.exit(0 if success else 1)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        print_usage()
        sys.exit(1)
