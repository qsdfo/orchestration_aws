# Orchestration_AWS

Automatic orchestration plug-in.
Python server deployed on AWS with a Max4live device client.

## Instructions
On Client:

    ssh -R 5005:localhost:5005 -I "*.pem" amazon_address
    socat tcp4-listen:5005,reuseaddr,fork UDP:localhost:5002 &

Then, on AWS:

    conda deactivate
    socat -T15 udp4-recvfrom:5002,reuseaddr,fork tcp:localhost:5005 &
    cd code/orchestration_aws
    source venv/bin/activate
    cd Max_server
    python osc_launch.py --ip=0.0.0.0 --ip_client=127.0.0.1 --in_port=5001 --out_port=5002
    
Back on client, use ableton live session

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