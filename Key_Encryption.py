from Crypto.Cipher import AES #This is the library
import pickle

def Encrypt(key,original_message):
    
    #key = b'gavecrtqogavecrtqogavecrtqo23fde' # 32 bytes key
    print("type of key", type(key))
    #b converts the string content to bynary
    cipher = AES.new(key, AES.MODE_EAX) #MODE_EAX => is about "how" to encript
    data = str.encode(original_message) # we need to convert to bytes - we encrypt bytes - not strings
    print("type of data", type(data))
    ciphertext, messageDigest = cipher.encrypt_and_digest(data)
    #chiperText contains the full content cryptographed
    #hash contains the hash of original content for later validation
    print("cryptographed content", ciphertext, type(ciphertext), len(ciphertext))
    print("HASH", messageDigest, type(messageDigest), len(messageDigest))

    print(cipher)
    nonce = cipher.nonce
    return ciphertext, messageDigest,nonce


# Read the authtoken from the text file
with open("./groq_api.txt") as f:
    original_message = f.read().strip()

#Its keys can be 128, 192, or 256 bits long.
key = b'09865rfqghlafgtz78nafg3q' # 24 bytes key        
ciphertext, messageDigest,nonce=Encrypt(key,original_message)

encrypted_data = {
    "ciphertext": ciphertext,
    "nonce": nonce,
    "message_digest": messageDigest
}

encrypted_filename = r"./encrypted_data.pkl"

with open(encrypted_filename, "wb") as pickle_file:
    pickle.dump(encrypted_data, pickle_file)

