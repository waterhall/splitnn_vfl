ssh-keygen -t rsa -f ssh_keys/$1.key -C darthvader_aka_eric
ssh-keygen -p -m PEM -f ./ssh_keys/$1.key
