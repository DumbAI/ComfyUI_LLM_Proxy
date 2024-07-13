import subprocess
import pkg_resources
def install_package(package_name):
    try:
        # Check if the package is already installed
        pkg_resources.get_distribution(package_name)
        print(f"{package_name} is already installed.")
    except pkg_resources.DistributionNotFound:
        # If not installed, install the package using pip
        print(f"Installing {package_name}...")
        subprocess.run(["pip", "install", package_name], check=True)
        print(f"{package_name} installed successfully.")


for package in ['pydantic', 'instructor', 'openai', 'opencv-python']:
    install_package(package)




from .llm_proxy import LLMProxy

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LLMProxy": LLMProxy
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMProxy": "LLMProxy Node"
}




__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]