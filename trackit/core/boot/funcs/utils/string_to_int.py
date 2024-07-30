import hashlib


def string_to_int_sha256(unique_string: str):
    hash_object = hashlib.sha256(unique_string.encode())
    hex_dig = hash_object.hexdigest()
    int_value = int(hex_dig, 16)

    return int_value
