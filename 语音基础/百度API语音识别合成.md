# 创建百度应用程序

[创建百度应用获取秘钥](https://ai.baidu.com/ai-doc/SPEECH/qknh9i8ed)

**创建应用**

- 您需要创建应用才可正式调用语音技术能力，应用是您调用服务的基本操作单元，您可以基于应用创建成功后获取的`API Key`及`Secret Key`，进行接口调用操作，及相关配置。

	

**示例**

[百度应用管理中心](https://console.bce.baidu.com/ai/?_=1678519931378#/ai/speech/app/list)

# ![image-20230311153349980](C:\Users\23606\AppData\Roaming\Typora\typora-user-images\image-20230311153349980.png)



---

# 调用百度 API

[短语音识别百度官方教程](https://ai.baidu.com/ai-doc/SPEECH/0lbxfnc9b)

[语音合成百度官方教程](https://ai.baidu.com/ai-doc/SPEECH/Slbxhog9b)

[百度语音SDK下载](https://ai.baidu.com/sdk#tts)

![image-20230311140732354](C:\Users\23606\AppData\Roaming\Typora\typora-user-images\image-20230311140732354.png)



**语音识别 Python SDK目录结构**

```text
├── README.md
├── aip                   //SDK目录
│   ├── __init__.py       //导出类
│   ├── base.py           //aip基类
│   ├── http.py           //http请求
│   └── speech.py //语音识别
└── setup.py              //setuptools安装
```

- 如果已安装pip，执行`pip install baidu-aip`即可。



---

## 新建 AipSpeech

AipSpeech是语音识别的Python SDK客户端，为使用语音识别的开发人员提供了一系列的交互方法。

```python
from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '你的 App ID'
API_KEY = '你的 Api Key'
SECRET_KEY = '你的 Secret Key'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
```

在上面代码中，常量`APP_ID`在百度云控制台中创建，常量`API_KEY`与`SECRET_KEY`是在创建完毕应用后，系统分配给用户的，均为字符串，用于标识用户，为访问做签名验证。





---

# 语音识别及合成



## 录音文件`audio_record.py`

```python
import keyboard
import pyaudio,wave
from tqdm import tqdm

def record(filename):
    p =  pyaudio.PyAudio()      # 实例化pyaudio对象

    SECONDS = 5                 # 时长(秒)
    FORMAT = pyaudio.paInt16    # 音频格式，即采样位深
    CHANNELS = 1                # 通道数
    RATE = 16000               	# 采样率
    CHUNK = 1024                # 采样帧
    OUTPUT_FILE = filename      #输出的录音文件

    # 打开音频流准备开始录制
    stream = p.open(rate=RATE,
                    format=FORMAT,
                    channels=CHANNELS,
                    input=True,                 # 代表此时是输入音频流
                    frames_per_buffer=CHUNK)    # 缓冲区大小为一帧
    
    frames = []
    frame_num = int(RATE * SECONDS / CHUNK)    	# 采样帧个数
    
    print("按下空格键开始录制音频!") 
    while True:
        if keyboard.is_pressed(' '):
            break
        
    print("录制中...")       
    for i in tqdm(range(frame_num)):            #tqdm显示进度条
        data = stream.read(CHUNK)       
        frames.append(data)

    stream.stop_stream()    # 停止采集
    stream.close()          # 关闭音频流
    p.terminate()           # 关闭pyaudio
    
    wf = wave.open(OUTPUT_FILE, 'wb')   # 打开文件

    # 设置参数
    wf.setframerate(RATE)
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 采样位深(字节)

    # frames为二进制列表文件,数据个数为采样帧数,这里把数据连续写入
    wf.writeframes(b''.join(frames))    
    wf.close()
```



---

## 播放文件`audio_play.py`

`os.system(cmd)`可以像终端一样调用系统命令

```python
import os

def play(file_name):
    """audio play"""
    os.system(f"ffplay -autoexit {file_name}")		#播放完毕自动退出
```



---

## 百度接口`baidu_ai.py`

```python
from aip import AipSpeech

APP_ID = '31151205'
API_KEY = 'e4FjjcWCVRalcCzQlbIe54ax'
SECRET_KEY = 'HmnVGM3CcLssQqtkTjK38R9aL0EvR6ah'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def audio_to_text(wav_file):
    """audio to test

    Args:
        wav_file (str): File to be converted

    Returns:
        str: file context
    """
    #读取音频文件
    with open(wav_file, 'rb') as fp:
        file_context = fp.read()
    
    # 识别本地文件
    res = client.asr(file_context, 'wav', 16000, {'dev_pid': 1537})     #res为字典类型
    res_str = res['result'][0]

    return res_str

def text_to_audio(synth_file, res_str):

    # 准备语音合成
    synth_context = client.synthesis(res_str,'zh',1,{
        'spd' : 5,  #语速(0-9)
        'vol' : 5,  #音量(0-9)
        'pit' : 5,  #音调(0-9)
        'per' : 4,  #发音人：度丫丫
    })

    # 确定合成内容已生成，因为生成错误会返回字典类型报错
    if not isinstance(synth_context, dict):
        with open(synth_file, 'wb') as f:
            f.write(synth_context)
        return 0
    else:
        return -1
```



---

## 主程序 `main.py`

```python
from audio_record import record
from audio_play import play
from baidu_ai import audio_to_text, text_to_audio

file = 'test.wav'           # 语音录制，识别文件
synth_file = "synth.mp3"    # 语音合成文件 

record(file)                # 录制音频 

res_str = audio_to_text(file)               # 语音识别
print(res_str)                              # 打印识别结果
 
ret = text_to_audio(synth_file, res_str)    # 语音合成

if ret != -1:
    play(synth_file)                        # 播放合成结果

```

