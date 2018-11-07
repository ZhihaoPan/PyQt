import pathlib,os
import binascii

def is_exists(path):
    if path.exists():
        return True
    else:
        return False

def crc32asii(str):
    str=str.encode('GBK')
    return '0x%8x' % (binascii.crc32(str) & 0xffffffff)

def crc32hex(str):
    return '%08x' % (binascii.crc32(binascii.a2b_hex(str)) & 0xffffffff)

if __name__=="__main__":
    str='{"filename"="123"}'
    print(crc32asii(str))