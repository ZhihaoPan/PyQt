import pathlib,os
import binascii
import json

def is_exists(path):
    if path.exists():
        return True
    else:
        return False

def crc32asii(str):
    """
    用于字符串的CRC校验，字符串用GBK编码，返回校验结果字符串
    :param str:
    :return:
    """
    str=str.encode('GBK')
    return '0x%8x' % (binascii.crc32(str) & 0xffffffff)

def crc32hex(str):
    return '%08x' % (binascii.crc32(binascii.a2b_hex(str)) & 0xffffffff)

def getChstr(msg):
    """
    无论msg已经是json格式还是不是json格式都进行一次json.load
    !!!注意要求：json 语法规定 数组或对象之中的字符串必须使用双引号，不能使用单引号
    :param msg:
    :return:
    """
    #将字符串用json读入
    dictmsg=json.load(msg)
    dictmsg.pop("chsum")

if __name__=="__main__":
    str='{"filename":"123"}'.encode("GBK")
    print(json.load(str))