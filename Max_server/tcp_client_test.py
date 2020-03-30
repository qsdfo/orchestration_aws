import socket

HOST = '63.33.36.17'
PORT = 5003

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
print(f'Connexion vers {HOST}:{PORT} reussie')

message = 'Yo TCP'
print(f'Envoi de : {message}')
n = client.send(str.encode(message))
if n != len(message):
    print('Erreur envoi')
else:
    print('Envoi ok')

print('Reception...')
donnees = client.recv(1024)
print('Recu :', donnees)

print('Deconnexion')
client.close()
