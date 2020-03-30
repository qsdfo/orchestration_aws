import socket
ADRESSE = '0.0.0.0'
PORT = 5003

serveur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serveur.bind((ADRESSE, PORT))
serveur.listen(1)
client, adresseClient = serveur.accept()
print(f'Connexion de {adresseClient}')

donnees = client.recv(1024)
if not donnees:
    print(f'Erreur de reception')
else:
    print(f'Reception de: {donnees}')

    reponse = donnees.upper()
    print(f'Envoi de : {reponse}')
    n = client.send(reponse)
    if (n != len(reponse)):
        print(f'Erreur envoi')
    else:
        print(f'Envoi ok')

print(f'Fermeture de la connexion avec le client.')
client.close()
print(f'Arret du serveur.')
serveur.close()
