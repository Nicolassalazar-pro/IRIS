import importlib
import pkg_resources
import sys

def get_version(module_name):
    """Attempt to get version information for a module using various methods."""
    try:
        # Try to get the distribution version first (most reliable)
        try:
            return pkg_resources.get_distribution(module_name).version
        except pkg_resources.DistributionNotFound:
            pass
        
        # Try common version attributes
        module = importlib.import_module(module_name)
        for attr in ['__version__', 'version', 'VERSION', '_version_']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, str):
                    return version
                elif hasattr(version, '__str__'):
                    return str(version)
                
        # For sub-modules, try to get the parent package version
        if '.' in module_name:
            parent_module = module_name.split('.')[0]
            try:
                return pkg_resources.get_distribution(parent_module).version
            except pkg_resources.DistributionNotFound:
                pass
                
        return "Version not found"
    except Exception as e:
        return f"Error: {str(e)}"

# List of modules to check 

modules = [
    "watchdog",
    "watchdog.events",
    "watchdog.observers",
    "contextlib",
    "sklearn",
    "sklearn.cluster",
    "pymongo",
    "sounddevice",
    "pathlib",
    "face_recognition",
    "PIL",
    "numpy",
    "threading",
    "keyboard",
    "datetime",
    "hashlib",
    "logging",
    "pyttsx3",
    "whisper",
    "string",
    "random",
    "pickle",
    "ollama",
    "msvcrt",
    "queue",
    "torch",
    "json",
    "time",
    "wave",
    "pytz",
    "cv2",
    "sys",
    "os"
]

def main():
    print("\nModule Version Checker")
    print("====================\n")
    print(f"Python version: {sys.version}\n")
    print("Imported Modules:")
    print("-----------------")
    
    max_module_length = max(len(module) for module in modules)
    format_string = f"{{:<{max_module_length + 2}}}{{:<}}"
    
    for module_name in modules:
        try:
            version = get_version(module_name)
            print(format_string.format(module_name + ":", version))
        except ImportError:
            print(format_string.format(module_name + ":", "Module not installed or not found"))

if __name__ == "__main__":
    main()