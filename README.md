# CS702 Computational Interaction

This repository contains code for the lectures and assignments for the corresponding course, CS702 Computational Interaction.

* Instructor: Kotaro Hara (Assistant Professor, SMU)
* Course Website: [Link](https://smuhci.notion.site/CS702-Computational-Interaction-f6f6a99c2877403ba31009d4bea63406)


Instructions for Running the Code in this Repo
==============================================


Follow the steps below:
1. Install Git and Docker.
1. Install VS Code and the Dev Container extension. 
1. Clone this repository
1. Run `Dev Containers: Reopen in Container`
1. Install dependencies

Install Git and Docker
----------------------
Install Git if your computer does not have it installed.

Install Docker if your computer does not have it installed.
Download Docker Desktop from the official website (https://www.docker.com/products/docker-desktop). 
Then, follow the installation instruction for your operating system (Windows/macOS/Linux).
After installation, start Docker Desktop and make sure it is running.
You can check this by running `docker --version` in your terminal.


Install VS Code and the Dev Containers Extension
------------------------------------------------
Download and install Visual Studio Code from https://code.visualstudio.com.
Open VS Code and navigate to the Extensions marketplace (`Ctrl+Shift+X` for Windows or `Cmd+Shift+X` for Mac).
Search for "Dev Containers" in the marketplace.
Then, click "Install" on the Dev Containers extension by Microsoft.

Clone This Repository
---------------------
Open your terminal and navigate to the directory where you want to clone this repository.
Then, run the following command to clone the repository and change into the directory:

```bash
git clone https://github.com/SMU-HCI-Lab/cs702-ci.git
cd cs702-ci
```

Run Dev Containers
------------------
Open the cloned project in VS Code. 
Press `Ctrl + Shift + P` for Windows or `Cmd + Shift + P` for macOS to open the command palette.
Type `Dev Containers: Reopen in Container` and select it.
Wait for VS Code to build and start the development container.
This process may take several minutes during the first run.

Install Dependencies
--------------------
Once inside the container, open a new terminal in VS Code.
Run the following command to install project dependencies:
```bash
uv pip install -e .
```