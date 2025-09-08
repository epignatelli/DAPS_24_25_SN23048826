import os

def decrypt(text):
    """
    Decrypts the encrypted username and password
    Vigniere cipher with a fixed key
    """
    key = "daps"
    result = []
    key_index = 0
    for ch in text:
        if 'a' <= ch <= 'z':
            shift = ord(key[key_index % len(key)]) - ord('a')
            result.append(chr((ord(ch) - ord('a') - shift) % 26 + ord('a')))
            key_index += 1
        elif 'A' <= ch <= 'Z':
            shift = ord(key[key_index % len(key)]) - ord('a')
            result.append(chr((ord(ch) - ord('A') - shift) % 26 + ord('A')))
            key_index += 1
        else:
            result.append(ch)
    return "".join(result)


def create_directory(name):
    """
    Create directory under the current path   
    """
    current_path = os.getcwd()
    path = os.path.join(current_path, name)
    if not os.path.isdir(path):
        os.mkdir(path)
