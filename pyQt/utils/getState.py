import psutil
import time
import pynvml

def getCPUstate(intervla=1):
    """
    pip install psutil
    :param intervla: 百分比
    :return: 字符串的cpu百分比使用率
    """
    return "CPU使用率为："+str(psutil.cpu_percent(intervla))+"%"

def getCurrentTime():
    return time.asctime(time.localtime(time.time()))

def getMemorystate():
    try:
        phymem=psutil.virtual_memory()
        line = "内存使用率为%5s%%  %6s(Used) / %s(Total)" % (
            phymem.percent, str(int(phymem.used / 1024 / 1024)) + "M", str(int(phymem.total / 1024 / 1024)) + "M")
        return line
    except Exception as e:
        return "出现错误 Error:"+str(e)

def getTemstate():
    """
    只可以再linux或者macos下使用，windows上没有这个功能
    :return:
    """
    # if hasattr(psutil,"sensors_temperatures"):
    #     tem=psutil.sensors_temperatures(fahrenheit=False)
    # else:
    #     return 0
    # return tem
    try:
        tem=psutil.sensors_temperatures(fahrenheit=False)
        tem_str="Temperatures of each Devices||\n"
        for name,entries in tem.items():
            for entry in entries:
                tem_str+="label:"+entry.label+" ,Tem:"+str(entry.current)+" |\n"
        return tem_str
    except Exception as e:
        return "出现错误 Error:"+str(e)

def getGPUstate():
    """
    pip install nvidia-ml-py3
    :return:返回一个数组，数组长度为GPU的个数
    """
    meminfo={}
    infoStr=""
    try:
        pynvml.nvmlInit()
        devicecount=pynvml.nvmlDeviceGetCount()
        for num in range(devicecount):
            handle=pynvml.nvmlDeviceGetHandleByIndex(num)
            info=pynvml.nvmlDeviceGetMemoryInfo(handle)
            meminfo[num]="Device: {} , {} / {} {:.2f}%, free memory:{}".format(num,info.used,info.total,info.used/info.total*100,info.free)
        for i in range(len(meminfo)):
            infoStr+=meminfo[i]+"\n"
        return infoStr
    except Exception as e:
        #print("error happen in getGPUstate:"+str(e))
        return "出现错误 Error:"+str(e)

if __name__=="__main__":
    meminfo=getTemstate()
    print(meminfo)
#    print(meminfo)
    #print(getGPUstate())