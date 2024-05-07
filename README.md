## Setup Instructions
1. Install Python [here](https://www.python.org/downloads/). You can find more detailed instructions [here](https://www.digitalocean.com/community/tutorials/install-python-windows-10).
    
    a. Verify by opening the Powershell terminal and running the following command. If installed and set up correctly, the corresponding python version should appear as output.

    ```
    python -V
    ```
2. Open a Powershell terminal on your Windows computer and move into the directory you installed this github repository.

   ```
    cd [PATH_TO_LOCAL_DIRECTORY]\Spacemouse_Test
    ```
3. Run the setup script. This will create a Python virtual environment named `manual_alignment_venv` with all the necessary python modules already installed inside.  
    ```
    ./setup.ps1
    ```

## Run Instructions

Let's run a basic manual alignment test using the 3Dconnexion Spacemouse Compact to see how intuitive it is!

1. Open a Powershell terminal on your Windows computer and move into the directory you installed this github repository.

2. Activate your Python virtual environment
    ```
    .\manual_alignment_venv\Scripts\Activate.ps1
    ```
3. Run the main.py script.
    ```
    python .\src\main.py
    ```
4. To exit the window, press the ESC key.


