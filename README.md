# Orchestration_AWS

Automatic orchestration plug-in.
Python server deployed on AWS with a Max4live device client.

## Instructions
### Get access to AWS
- Log in AWS dashboard
- Click on *EC2* > *pair of keys* > *Create a new pair*
- Choose a name for the key, create it and **download the .pem file** somewhere you can easily find it.
- Open a terminal (CMD+tab > terminal)
    - cd to the location of your file (mine was in Downloads, so I typed:)
            
            cd ; cd Downloads
    - replacing *name_pem* with the name of your .pem file, type
        
            mv name.pem  ~/.ssh/
            chmod 400 ~/.ssh/name.pem
            ssh-keygen -y -f ~/.ssh/name.pem
- Send me (crestel.leopold@gmail.com) the public key which has been displayed on your screen (just copy/paste) 
and wait till I granted you access to the AWS server

### Create an SSH tunnel
Once first step is done, open a terminal (CMD + terminal) on your computer:

    socat tcp4-listen:5005,reuseaddr,fork UDP:localhost:5002 &
    ssh -R 5005:localhost:5005 -i "~~.ssh/name.pem" amazon_address

That should have openned an ssh tunnel between AWS and your computer.
You can now use the plug-in in an ableton live session on your computer.
Just check in the plug-in that send port is 5001, receive port 5002, and ip send 63.33.36.17, 
but these shoud be the default values 

**If it does not work**, contact me and we'll have to try the version where you launch the server by yourself 

    conda deactivate
    socat -T15 udp4-recvfrom:5002,reuseaddr,fork tcp:localhost:5005 &
    cd code/orchestration_aws
    source venv/bin/activate
    cd Max_server
    python osc_launch.py --ip=0.0.0.0 --ip_client=127.0.0.1 --in_port=5001 --out_port=5002

## Improving server
- GNU Socket options
https://www.gnu.org/software/libc/manual/html_node/Socket_002dLevel-Options.html#Socket_002dLevel-Options
- Python socket lib
https://docs.python.org/2/library/socket.html
- Pour une version plus légère dans le future, n'utilisant que socket 
http://python.jpvweb.com/python/mesrecettespython/doku.php?id=client-serveur_udp_mini
- Minimalist UDP communication example
https://wiki.python.org/moin/UdpCommunication
- Exemples client TCP: https://fiches-isn.readthedocs.io/fr/latest/FicheReseauxClient01.html
- Exemples server TCP: https://fiches-isn.readthedocs.io/fr/latest/FicheReseauxServeur01.html