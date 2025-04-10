# IRIS Project Setup and Troubleshooting Guide

This guide provides step-by-step instructions on how to set up the IRIS project from scratch, including troubleshooting common issues that may arise during installation and setup.

## Prerequisites

Before starting the setup, ensure the following are installed:

* Node.js (for running frontend and server)
* npm (Node package manager)
* Python 3.x (for backend services, e.g., Flask)
* pip (Python package manager)
* CMake (required for building certain dependencies like dlib)

Additionally, if you are working with GPU support (CUDA), ensure the correct version of PyTorch and its dependencies are installed. Visit [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) Pytorch build: Stable, Package: Pip, Your OS: #your machine, Compute Platform: CUDA 11.8 Then run:

```bash
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
Step 1: Clone or Fork the Repository
Cloning the Repository
To clone the repository:

Open your terminal and run the following command (replace the URL with the actual repository URL):

Bash

git clone [https://github.com/username/repo-name.git](https://github.com/username/repo-name.git)
Navigate to the project directory:

Bash

cd repo-name
Forking the Repository (Optional)
If you plan to contribute or maintain your own version, you may want to fork the repository. Follow these steps:

Go to the GitHub repo and click the Fork button (top-right corner).

Clone your forked repository:

Bash

git clone [https://github.com/your-username/your-forked-repo.git](https://github.com/your-username/your-forked-repo.git)
Set the remote URL to your fork:

Bash

git remote set-url origin [https://github.com/your-username/your-forked-repo.git](https://github.com/your-username/your-forked-repo.git)
Confirm the change:

Bash

git remote -v
Step 2: Install Backend Dependencies (Python)
Install Python Dependencies
Navigate to the backend directory (usually V3 in this case, if not check root directory).

Install the required Python dependencies by running:

Bash

pip install -r requirements.txt
If a specific package is missing, such as flask_cors, install it using pip:

Bash

pip install flask-cors
If you see errors related to missing packages like sounddevice, numpy, or pyttsx3, install them using the following command:

Bash

pip install sounddevice numpy pyttsx3 whisper ollama torch flask flask-cors pytz requests keyboard
Fixing Config File Path Issues:
If you encounter a path error like:

JavaScript

Error loading config: [Errno 2] No such file or directory: 'A:\\IRIS\\V3\\config.py'
Update the path in your code to point to the correct location:

Python

CONFIG_FILE = "C:\\Users\\Ikean\\IRIS\\V3\\config.py"
# Change to your relative path
Step 3: Install Frontend Dependencies (Node.js)
Navigate to the UI directory of the project.

Install all required Node.js dependencies:

Bash

npm install
To start the frontend application, run:

Bash

npm start
If you encounter the error ENOENT: no such file or directory, stat 'C:\Users\Ikean\IRIS\UI\static', this means the static directory is missing. You can create it:

Navigate to C:\Users\Ikean\IRIS\UI (to your own relative path).

Create a static directory.

After creating the directory, you can add any necessary static assets (images, JavaScript files, etc.). Copy and paste the src folder into the newly created static folder. Also ensure folder name is "static" not "Static".

If the build process fails with errors related to missing static files, ensure your build configuration files (like parcel-config.js) are set correctly to reference the new static directory. #This step might not be necessary should automatically reference the static folder

Step 4: Install CMake (If Required)
If you see errors related to dlib needing CMake, you'll need to install it:

Download CMake from the official CMake website.

During installation, ensure that you select the option to Add CMake to the system PATH.

After installation, verify that CMake is correctly installed:

Bash

cmake --version
Step 5: Start the Full Application
Now that you've set up both the frontend and backend, you can start the application:

Run the full application using npm start:

Bash

npm start
This command will run all parts of the application concurrently (frontend, Flask backend, and server).

The following services will be started:

Frontend: Accessible via http://localhost:8008
Backend (Flask): Running on the specified port
Server: Running at https://localhost:6969
Common Errors and Fixes
Missing Python Modules: If you encounter errors like ModuleNotFoundError, ensure the necessary Python packages are installed:

Bash

pip install flask flask-cors sounddevice numpy pyttsx3 whisper ollama torch pytz requests keyboard
Static Directory Missing: If you see an error related to the missing static directory, create it manually in the UI folder. Add any necessary static files (images, JS, etc.) to this directory.

CMake Not Found: If dlib installation fails due to a missing CMake installation, follow the CMake installation steps and make sure CMake is in your system PATH.

PyTorch Installation Issues: If you're getting errors related to installing torch, ensure you're installing the correct version of PyTorch that matches your Python version and CUDA setup (if using a GPU). To install PyTorch with CUDA 11.8 support, run:

Bash

pip install torch==2.6.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.1 -f [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
Conclusion
By following this guide, you should be able to set up the IRIS project on your local machine and troubleshoot any common issues. If you continue to face problems, consider checking the project’s GitHub issues page or contacting the project maintainers for additional help.