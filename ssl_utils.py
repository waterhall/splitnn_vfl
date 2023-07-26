from OpenSSL import crypto
import os
import ssl
from remote import File
import copy

TYPE_RSA = crypto.TYPE_RSA
TYPE_DSA = crypto.TYPE_DSA

def create_key_pair(type, bits):
   pkey = crypto.PKey()
   pkey.generate_key(type, bits)
   return pkey

def create_cert_request(pkey, digest="md5", **name):
   req = crypto.X509Req()
   subj = req.get_subject()

   for (key, value) in name.items():
       setattr(subj, key, value)

   req.set_pubkey(pkey)
   req.sign(pkey, digest)
   return req

def createCertificate(req, issuerCertKey, serial, validityPeriod,
                  digest="sha256"):
    issuerCert, issuerKey = issuerCertKey
    notBefore, notAfter = validityPeriod
    cert = crypto.X509()
    cert.set_serial_number(serial)
    cert.gmtime_adj_notBefore(notBefore)
    cert.gmtime_adj_notAfter(notAfter)
    cert.set_issuer(issuerCert.get_subject())
    cert.set_subject(req.get_subject())
    cert.set_pubkey(req.get_pubkey())
    cert.sign(issuerKey, digest)
    return cert

class SSHKeyPairFile(File):
    def __init__(self, key_ref):
        super().__init__()
        self.key_ref = key_ref
        # This means key is not init
    def file_names(self):
        return [ f'{self.key_ref}.key.pem', f'{self.key_ref}.cert.pem' ]
    def create(self, exp_path):
        super().create(exp_path)        
        pri_key_path, pub_key_path = self.file_paths(exp_path)
        
        cakey = create_key_pair(TYPE_RSA, 1024)
        careq = create_cert_request(cakey, CN='Certificate Authority')
        cacert = createCertificate(careq, (careq, cakey),0, (0, 60 * 60 * 24 * 365))  # one year

        open(pri_key_path, 'wb').write(crypto.dump_privatekey(crypto.FILETYPE_PEM, cakey))
        open(pub_key_path, 'wb').write(crypto.dump_certificate(crypto.FILETYPE_PEM, cacert))

    def transfer(self, exp_path, key_ref=None):
        copied_ssh_pair_file = copy.deepcopy(self)
        copied_ssh_pair_file.exp_path = exp_path
        if key_ref != None:
            copied_ssh_pair_file.key_ref = key_ref
        return copied_ssh_pair_file
    def ssl_context(self):
        if self.exp_path == None:
            raise EnvironmentError
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        pri_key_path, pub_key_path = self.file_paths(self.exp_path)
        ssl_context.load_cert_chain(pub_key_path, pri_key_path)
        return ssl_context
