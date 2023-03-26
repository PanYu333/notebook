# 用pyaudio录制音频和播放



## 录制音频文件


```python
import pyaudio,wave

p =  pyaudio.PyAudio()      # 实例化pyaudio对象

SECONDS = 8                 # 时长(秒)
FORMAT = pyaudio.paInt16    # 音频格式，即采样位深
CHANNELS = 1                # 通道数
RATE = 16000               	# 采样率
CHUNK = 1024                # 采样帧
OUTPUT_FILE = "test.wav"    #输出的录音文件

# 打开音频流准备开始录制
stream = p.open(rate=RATE,
                format=FORMAT,
                channels=CHANNELS,
                input=True,                 # 代表此时是输入音频流
                frames_per_buffer=CHUNK)    # 缓冲区大小为一帧

print("录音开始！")
frames = []

frame_num = int(RATE * SECONDS / CHUNK)    	# 采样帧个数

for i in range(frame_num):
    data = stream.read(CHUNK)       
    frames.append(data)

stream.stop_stream()    # 停止采集
stream.close()          # 关闭音频流

wf = wave.open(OUTPUT_FILE, 'wb')   # 打开文件

# 设置参数
wf.setframerate(RATE)
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))  # 采样位深(字节)

# frames为二进制列表文件,数据个数为采样帧数,这里把数据连续写入
wf.writeframes(b''.join(frames))    
wf.close()

```





## 播放音频文件

```python
"""播放音频文件"""
wf = wave.open('test.wav', mode='rb')

stream = p.open(rate=wf.getframerate(),
                channels=wf.getnchannels(),
                format=p.get_format_from_width(wf.getsampwidth()),
                output=True)

print("开始播放音频！")

# 读取数据
data = wf.readframes(CHUNK)

# 播放  
while data != b'':
    stream.write(data)              #从流对象播放
    data = wf.readframes(CHUNK)     #继续写入

# 停止数据流  
stream.stop_stream()
stream.close()

# 关闭pyaudio
p.terminate()
```





## 加入按键控制的语音录制和播放

```python
import pyaudio,wave
import keyboard


def record(rate, seconds, chunk):
    """record function

    Args:
        rate (scalar): sample rate
        seconds (scalar): record time
        chunk (scalar): Sample frame

    Returns:
        frames: a list of binary audio data
    """
    frames = []

    frame_num = int(rate * seconds / chunk)    # 采样帧个数

    print("录制中...")
    for i in range(frame_num):
        data = stream.read(chunk)       
        frames.append(data)
    
    print("录制完毕!")
        
    return frames

def audio_write(file, frames, rate, channels, format):
    """write audio data

    Args:
        file (str): File to be write
        frames (list): a list of binary audio data
        rate (scalar): sample rate
        channels(scalar): audio channels
        seconds (scalar): record time
    """
    wf = wave.open(file, 'wb')   # 打开文件

    # 设置参数
    wf.setframerate(rate)
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))  # 采样位深(字节)

    # frames为二进制列表文件,数据个数为采样帧数,这里把数据连续写入
    wf.writeframes(b''.join(frames))   
    print("数据写入完毕!") 
    wf.close()
   
    
def audio_play(file, chunk):
    """audio play

    Args:
        file (str): audio file
        chunk (scalar): sample frame
    """
    wf = wave.open(file, mode='rb')

    stream = p.open(rate=wf.getframerate(),
                    channels=wf.getnchannels(),
                    format=p.get_format_from_width(wf.getsampwidth()),
                    output=True)

    print("开始播放音频！")
    print("播放中...")
    
    # 读取数据
    data = wf.readframes(chunk)

    # 播放  
    while data != b'':
        stream.write(data)              #从流对象播放
        data = wf.readframes(chunk)     #继续写入

    print("播放完毕!")
    # 停止数据流  
    stream.stop_stream()
    stream.close()
   
    
p =  pyaudio.PyAudio()      # 实例化pyaudio对象

SECONDS = 5                 # 时长(秒)
FORMAT = pyaudio.paInt16    # 音频格式，即采样位深
CHANNELS = 1                # 通道数
RATE = 16000                # 采样率
CHUNK = 1024                # 采样帧
OUTPUT_FILE = "output.wav"  #输出的录音文件


"""录制音频文件"""
# 创建音频流对象
stream = p.open(rate=RATE,
                format=FORMAT,
                channels=CHANNELS,
                input=True,                 # 代表此时是输入音频流
                frames_per_buffer=CHUNK)    # 缓冲区大小为一帧

print("按下空格键开始录制音频")

while True:
    if keyboard.is_pressed(' '):
        print("录音开始！")
        break
#获取到音频流数据
frames = record(rate=RATE, seconds=SECONDS, chunk=CHUNK)

stream.stop_stream()    # 停止采集
stream.close()          # 关闭音频流

#写入音频数据
audio_write(OUTPUT_FILE, frames, RATE, CHANNELS, FORMAT)


"""播放音频文件"""
audio_play(OUTPUT_FILE, chunk=CHUNK)

# 关闭pyaudio
p.terminate()
```

