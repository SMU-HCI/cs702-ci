# CS702 Computational Interaction

This repository contains code for the lectures and assignments for the corresponding course, CS702 Computational Interaction.

* Instructor: Kotaro Hara (Assistant Professor, SMU)
* Course Website: [Link](https://smuhci.notion.site/cs702-ci)


Instructions for Running the Code in this Repo
==============================================


Follow the steps below:
1. Install Git and PIXI
1. Install VS Code
1. Clone this repository
1. Install dependencies


Install Git and PIXI
----------------------
Install Git if your computer does not have it installed.

Install PIXI if your computer does not have it installed.
Follow the instruction here (https://pixi.prefix.dev/latest/installation/)


Install VS Code
------------------------------------------------
Download and install Visual Studio Code from https://code.visualstudio.com.


Clone This Repository
---------------------
Open your terminal and navigate to the directory where you want to clone this repository.
Then, run the following command to clone the repository and change into the directory:

```bash
git clone https://github.com/SMU-HCI-Lab/cs702-ci.git
cd cs702-ci
```

Install Dependencies
--------------------
Once inside the container, open a new terminal in VS Code.
Run the following command to install project dependencies:

```bash
pixi install
pixi run python -m pip install stormpy==1.11.3
```

Stormpy is installed separately because its macOS wheels require a newer
deployment target than Pixi uses while resolving the cross-platform lock file.
Using `python -m pip` ensures that it is installed into the Pixi environment
rather than another `pip` installation on your `PATH`.


Running Python in PIXI
----------------------
Run a command in the env:
```bash
pixi run python
```

Start a shell with the Python env active. This is most likely what you need to do.
```bash
pixi shell
```

If you want to run a single command without entering a shell, run:
```bash
pixi run python my_script.py
```


Jupyter Configuration
---------------------
To run Jupyter Notebook, run:

```bash
pixi run register-kernel
```

Then run run `Cmd+Shift+P` and `Developer: Reload Window`.

If you want to run Jupyter Notebook in VS Code, select pixi's Python as the VS Code interpreter.