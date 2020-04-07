# Orchestration_AWS

Automatic orchestration plug-in.
Python server deployed on AWS with a Max4live device client.

## Instructions
### Get authorisation to communicate with AWS instance
- Get your public IP (https://whatismyipaddress.com/fr/mon-ip). 
This is not necessary if you are on Sony network or VPN
- send me your public IP (crestel.leopold@gmail.com) and wait for my confirmation

### First time installation
A python script will be ran on your computer to interface AWS instance and Ableton.
Open a terminal (CMD + space: terminal).
Go to the location of your choice on your computer

    cd location
    
I would recommend creating a folder dedicated to the project

    mkdir orchestration
    cd orchestration

Check that you have python3 installed
    
    python3 --version
    
If not, follow the instructions here to install it (and don't hesitate to contact me): https://docs.python-guide.org/starting/install3/osx/
        
If you don't have virtualenv (you can check by typing "virtualenv --version" in the terminal) do

    python -m pip install --user virtualenv
        
Create a virtualenv (prevent python modules that will be installed to overwrite previous intallations).

    virtualenv venv -p python3

Then

    source venv/bin/activate
    pip install python-osc

Now, in the folder orchestration you created, 
open a file called tcp_udp_interface.py and copy the following Python script (https://github.com/qsdfo/orchestration_aws/blob/master/Max_server/tcp_udp_interface.py).
You can click on 'Raw' to copy/paste easily.
Installation is done!

### To be done everytime you want to use the plug-in
Go to the location of the orchestration folder you created.
    
    cd path-to-orchestration/orchestration
    
Activate the virtualenv

    source venv/bin/activate
    
Launch the intface

    python tcp_udp_interface.py

You can now use the plug-in in an ableton live session on your computer. 
Just check in the plug-in that send port is 5002 and receive port 5003.
    
