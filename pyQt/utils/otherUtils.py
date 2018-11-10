import pathlib,os
import binascii
import json

def is_exists(path):
    if path.exists():
        return True
    else:
        return False

def crc32asii(msg):
    """
    用于字符串的CRC校验，字符串用GBK编码，返回校验结果字符串
    :param str:
    :return:str
    """
    if isinstance(msg,str):
        str1=msg.encode('GBK')
        return '0x%8x' % (binascii.crc32(str1) & 0xffffffff)
    elif isinstance(msg,dict):
        msg=str(msg)
        str1 = msg.encode('GBK')
        return '0x%8x' % (binascii.crc32(str1) & 0xffffffff)


def crc32hex(str):
    return '%08x' % (binascii.crc32(binascii.a2b_hex(str)) & 0xffffffff)

def getChstr(msg):
    """
    首先判断传入的msg是不是json类型
    !!!注意要求：json 语法规定 数组或对象之中的字符串必须使用双引号，不能使用单引号
    :param msg:
    :return:
    """
    if isinstance(msg,dict):
        msg.pop("chsum")
        return msg
    elif isinstance(msg,str):
        dictmsg=json.loads(msg)
        dictmsg.pop("chsum")
        return dictmsg
    else:
        return None


if __name__=="__main__":
    # str1={"filename":"123"}
    # str2 = {"filename1": "123"}
    # # str2=""+str(str1)
    # # print(str2)
    # #print(json.loads(str))
    # dic={}
    # dic.update({"a":1})
    # print(dic)
    import zmq

    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to hello world server…")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    #  Do 10 requests, waiting each time for a response
    print("Sending request %s …" % 1)
    dic = {"head": "cmd", "file": r"D:\Dataset\Gunshot", "func_ycsyjc": 0, "func_yzfl": 0, "func_swfl": 0,
        "chsum": "0xf53b5ae7"}

    socket.send_json(dic)
    #  Get the reply.
    message = socket.recv_json()
    print("Received reply %s [ %s ]" % (1, message))


    socket=context.socket(zmq.REP)
    socket.bind("tcp://*:5556")
    message=dict(socket.recv_json())
    print("receive request:%s" % str(message))

    dic={
    "head":"report", "file":" / xxxx / xxxxx", "stop":0,    "chsum":"00000"}
    socket.send_json(dic)
