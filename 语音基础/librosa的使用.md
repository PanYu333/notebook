# librosa 概述

[librosa官方文档](https://librosa.org/doc/latest/index.html)

本文中常用的专业名词：

- **`sr`**：采样率

- **`hop_length`**：帧移
- **`overlapping`**：连续帧之间的重叠部分
- **`n_fft`**：窗口大小， 
	- **`n_fft`** = **`hop_length`** + **`overlapping`**
- **`win_length`** = **`frame`**: 窗长，一帧由窗长截取， 再添零去匹配`n_fft`，**默认`win_length = n_fft`**
- **`spectrum`**：频谱
- **`spectrogram`**：频谱图或叫做语谱图
- **`amplitude`**：振幅
- **`mono`**：单声道
- **`stereo`**：立体声





==**线性谱**、**梅尔谱**、**对数谱**：经过FFT变换后得到语音数据的线性谱，对线性谱取Mel系数，得到梅尔谱；对线性谱取对数，得到对数谱。==









---

# 时域

## 读取音频

```python
librosa.load(path, sr=22050, mono=True, offset=0.0, duration=None)
```

**参数**：

- **path** ：音频文件的路径。
- **sr** ：      采样率，如果为“None”使用音频自身的采样率
- **mono** ：bool，是否将信号转换为单声道
- **offset** ：float，在此时间之后开始阅读（以秒为单位）
- **duration**：float，仅加载这么多的音频（以秒为单位）

**返回：**

- **y** ：音频时间序列
- **sr**：音频的采样率



## 重采样

```python
librosa.resample(y, orig_sr, target_sr, fix=True, scale=False) 
```

重新采样从orig_sr到target_sr的时间序列

**参数**：

- **y** ：音频时间序列。可以是单声道或立体声。
- **orig_sr** ：y的原始采样率
- **target_sr** ：目标采样率
- **fix**：bool，调整重采样信号的长度，使其大小恰好为  $len(y) / orig\_sr * target\_s$
- **scale**：bool，缩放重新采样的信号，以使`y`和`y_hat`具有大约相等的总能量。

**返回**：

- **y_hat** ：重采样之后的音频数组





## 读取时长

```python
librosa.get_duration(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, center=True, filename=None)
```

计算时间序列的的**持续时间**（以秒为单位）

**参数**：

- **y** ：音频时间序列

- **sr** ：*y的*音频采样率

- **S** ：STFT矩阵或任何STFT衍生的矩阵（例如，色谱图或梅尔频谱图）。根据频谱图输入计算的持续时间仅在达到帧分辨率之前才是准确的。如果需要高精度，则最好直接使用音频时间序列。

- **n_fft** ：*S的* FFT窗口大小

- **hop_length** ：表示相邻窗之间的距离，这里为512，也就是相邻窗之间有75%的overlap

- ***center** ：*

	布尔值

	- 如果为True，则S [:, t]的中心为y [t * hop_length]
	- 如果为False，则S [:, t]从y[t * hop_length]开始

- **filename** ：如果提供，则所有其他参数都将被忽略，并且持续时间是直接从音频文件中计算得出的。

返回：

- **d** ：持续时间（以秒为单位）





## 读取采样率

```python
librosa.get_samplerate(path)
```

**参数**：

- **path** ：音频文件的路径

**返回**：音频文件的采样率



## 均方根能量

```python
librosa.feature.rms(y, frame_length=2048, hop_length=512, pad_mode='constant')
```

**公式**：值为第t帧中每点幅值平方再取均值后开根号

**返回**： ndarray, 	shape(1, frame_num)





## 过零率

```
librosa.feature.zero_crossing_rate(y, frame_length = 2048, hop_length = 512, center = True) 
```

每帧中语音信号从正变为负，或从负变为正的次数，一般情况下，过零率越大，频率越高

**应用**：语音识别，音乐信息检索

![过零率](C:\Users\23606\AppData\Roaming\Typora\typora-user-images\image-20230312220042530.png "过零率")



```python
"""
	手写版本(函数体中求当前帧过零率)：
		公式=该帧信号的平方和，开根号后，取帧长的平均值
		a = np.sign(current_frame[0:frame_length-1,])
        b = np.sign(current_frame[1:frame_length,]) 
		current_zcr = np.sum(np.abs(np.sign(a)-np.sign(b)))/2/frame_length
		其他内容与综合部分AE，RMSE几乎相同
"""
```



## 写音频

```python
import soundfile

soundfile.write(file, data, samplerate)
```

**参数：**

- **file**：保存输出wav文件的路径
- **data**：音频数据
- **samplerate**：采样率





## 绘制波形图

```python
librosa.display.waveshow(y, sr=22050, x_axis='time', offset=0.0)
```

绘制波形的幅度包络线， 点数达到 `sr` 为一秒

**参数**：

- **y** ：音频时间序列
- **sr** ：y的采样率
- **x_axis** ：str {'time'，'off'，'none'}或None，如果为“时间”，则在x轴上给定时间刻度线。
- **offset**：绘图起点，单位（s）





---

# 频域

## 谱质心`spectral_centroid`

是频率成分的重心，谱频率中一定范围内通过能量加权平均的频率。谱质心描述了声音的明亮度，具有阴暗、低沉品质的声音倾向有较多低频内容，谱质心相对较低，具有明亮、欢快品质的多数集中在高频，谱质心相对较高。该参数常用于对乐器声色的分析研究。

```python
librosa.feature.spectral_centroid(y, S, sr=22050, n_fft=2048, hop_length=512,window='hann', pad_mode='constant')
```

**公式**：sc = 对应点的频率f x 每一帧的频率幅值的和 / 每帧的频率幅值之和

**单位**：hz

<img src="https://img-blog.csdn.net/2018062709505117?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMjI5MjUz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom:80%;" />



## 子带带宽`bandwidth`

每一点到谱质心得加权平均值

```python
librosa.feature.spectral_bandwidth(y, sr, S, t_fft...)
```

**公式**：每个采样点减去谱质心的绝对值 x 对应的权重值 / 总的权重之和





## 幅值和相位

**librosa** 提供了专门将复数矩阵D(F, T)分离为幅值 $S$ 和相位 $P$ 的函数，$D = S * P$

```python
librosa.magphase(D, power=1)
```

**参数**：

- **D**：经过 stft 得到的复数矩阵
- **power**：幅度谱的指数，例如，1代表能量，2代表功率，等等。

**返回**：

- **D_mag**：幅值 $S$ ，
- **D_phase**：相位 $P$，



## 幅值转dB

```
librosa.amplitude_to_db(S, ref=1.0)
```

　　将幅度频谱转换为dB标度频谱。也就是**对S取对数**。与这个函数相反的是**`librosa.db_to_amplitude(S) `**

**参数**：

- **S** ：输入幅度
- **ref** ：参考值，幅值 abs(S) 相对于ref进行缩放

**返回**：

- 幅值，单位：dB



## 功率转dB

```
librosa.core.power_to_db(S, ref=1.0)
```

　　将功率谱(幅值平方)转换为dB单位，与这个函数相反的是 **`librosa.db_to_power(S)`** 

**参数**：

- **S**：输入功率
- **ref** ：参考值，振幅abs(S)相对于ref进行缩放，

**返回**：

- 功率谱，单位：dB





## 帧数转时间

```python
time_scale = librosa.frames_to_time(frame_scale, sr=22050, hop_length=512)
```

**参数**：

- **frames**：帧数
	- bool： 返回0或1帧时对应的时间
	- int/list/tupe/ndarray: 返回对应帧的时间，与输入同类型

- **sr** ：参考采样率
- **hop_length**: 帧移

**计算公式**： **`time[i] = frames[i] x hop_length / sr`**













---

## 短时傅里叶变换

```
librosa.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')
```

**参数**：

- **window**：字符串，元组，数字，函数 。shape =（n_fft, )
	- 窗口（字符串，元组或数字）；
	- 窗函数，例如`scipy.signal.hanning`
	- 长度为`n_fft`的向量或数组

- **center**：bool
	- 如果为True，则填充信号y，以使帧 D [:, t]以y [t * hop_length]为中心。
	- 如果为False，则D [:, t]从y [t * hop_length]开始

**返回：**

- `STFT`矩阵，$shape = (1 + n\_fft / 2, t)$

	

### 分帧加窗的具体操作

 首先要根据信号长度、帧移、帧长计算出该信号一共可以分的帧数，帧数的计算公式如下：

​									**帧数 = （信号长度-帧长）➗帧移 +1**

具体的分帧操作如下图所示：

![分帧加窗](https://img-blog.csdnimg.cn/57c650e7d42145f193d6826c7ab4d83d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA546L5bCP6L-QZQ==,size_20,color_FFFFFF,t_70,g_se,x_16 "分帧")



## 短时傅里叶逆变换

```python
librosa.istft(stft_matrix, hop_length=None, win_length=None, window='hann', center=True, length=None)
```

　　短时傅立叶逆变换（ISTFT），将复数值D(f,t)频谱矩阵转换为时间序列y，窗函数、帧移等参数应与stft相同

**参数**：

- **stft_matrix** ：经过STFT之后的矩阵
- **hop_length** ：帧移，默认为$win_length / 4$
- **win_length** ：窗长，默认为n_fft
- **window**: 窗类型
- **center**：bool
	- 如果为True，则假定D具有居中的帧
	- 如果False，则假定D具有左对齐的帧
- **length**：如果提供，则输出y为零填充或剪裁为精确长度音频

**返回**：

- **y** ：时域信号





---

## 绘制频谱图

```python
librosa.display.specshow(data, x_axis=None, y_axis=None, sr=22050, hop_length=512)
```

参数：

- **data**：要显示的矩阵
- **sr** ：采样率
- **hop_length** ：帧移
- **x_axis、y_axis**：时间x 和 频率y 轴的类型
- 时间类型
	- time：标记以毫秒，秒，分钟或小时显示。值以秒为单位绘制。
	- s：标记显示为秒。
	- ms：标记以毫秒为单位显示。
- 频率类型
	- 'linear'，'fft'，'hz'：频率范围由FFT窗口和采样率确定
	- 'log'：频谱以对数刻度显示
	- 'mel'：频率由mel标度决定

- 所有频率类型均以Hz为单位绘制





---

## Mel频谱

```
librosa.feature.melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)
```

如果提供了频谱图输入S，则通过mel_f.dot（S）将其直接映射到mel_f上。

如果提供了时间序列输入y，sr，则首先计算其幅值频谱S，然后通过mel_f.dot（S ** power）将其映射到mel scale上 。默认情况下，power= 2在功率谱上运行。

**参数**：

- **S**：频谱

- **power**：幅度谱的指数。例如1代表能量，2代表功率，等等
- **n_mels**：滤波器组的个数 1288
- **fmax**：最高频率

**返回**：Mel频谱shape=(n_mels, frame)



## Log-Mel Spectrogram

　　Log-Mel Spectrogram特征是目前在语音识别和环境声音识别中很常用的一个特征，由于CNN在处理图像上展现了强大的能力，使得音频信号的频谱图特征的使用愈加广泛，甚至比MFCC使用的更多。在librosa中，Log-Mel Spectrogram特征的提取只需几行代码：



```python
import librosa

y, sr = librosa.load('./train_nb.wav', sr=16000)
# 提取 mel spectrogram feature
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.power_to_db(melspec)        # 转换到对数刻度

print(logmelspec.shape)        # (128, 157)
```

可见，Log-Mel Spectrogram特征是二维数组的形式，128表示Mel频率的维度（频域），64为时间帧长度（时域），所以Log-Mel Spectrogram特征是音频信号的时频表示特征。其中，n_fft指的是窗的大小，这里为1024；hop_length表示相邻窗之间的距离，这里为512，也就是相邻窗之间有50%的overlap；n_mels为mel bands的数量，这里设为128。





---

## MFCC系数

　　MFCC特征是一种在自动语音识别和说话人识别中广泛使用的特征。

​		[梅尔频率倒谱系数(MFCC)的原理详解](https://www.cnblogs.com/LXP-Never/p/10918590.html)

```python
librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', **kwargs)
```

**参数**：

- y：音频数据
- sr：采样率
- n_mfcc：int>0，要返回的MFCC数量

- **非重点**：

	- S：np.ndarray，对数功能梅尔谱图

	- dct_type：None, or {1, 2, 3} 离散余弦变换（DCT）类型。默认情况下，使用DCT类型2。

	- norm： None or ‘ortho’ 规范。如果dct_type为2或3，则设置norm =’ortho’使用正交DCT基础。 标准化不支持dct_type = 1。

**返回**：

- M： MFCC序列，数组类型shape(norm, frame)





# 综合部分

## 时域振幅包络`AE`

<img src="C:\Users\23606\Pictures\dab05a64e26b425cae6e657a894fa816.png" alt="振幅包络公式" title="振幅包络公式" style="zoom:50%;" />

- 对每一帧求最大振幅，将每一帧最大值连起来就是幅值包络

**特点**:振幅包络给出响度`soundness`的大致信息，对突变信号敏感

**应用**：音频检测、音频分类



```python
"""
提取语音信号的频谱包络
    帧长：frame_length
    帧移：hop_length
    帧数：frame_num
    波形：wave_form
    添零数：pad_num
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

filename = './music_piano.wav'

# 加载音频文件, 返回数据和采样率
y, sr = librosa.load(path=filename, sr=None)


def cal_amplitude_envelope (waveform, frame_length, hop_length):
    if (len(waveform) - frame_length) % hop_length != 0:                    # 查看是否满足帧长
        frame_num = int((len(waveform) - frame_length) / hop_length) + 1    # 帧数
        pad_num = frame_num * hop_length + frame_length - len(waveform)     # 添零数
        waveform = np.pad(waveform, (0, pad_num), mode='wrap')              # 添零后的波形
    frame_num = int((len(waveform) - frame_length) / hop_length)            # 帧数(此时应该必然可以整除)

    # 每帧的最大值放在列表中
    waveform_ae = []
    for i in range(frame_num):
        # 取出每一帧，求最大值放入列表
        currect_frame = waveform[i*hop_length:i*hop_length + frame_length]
        currect_ae = max(currect_frame)
        waveform_ae.append(currect_ae)

    return np.array(waveform_ae)


frame_size = 1024
hop_size = int(frame_size / 4)

waveform_AE = cal_amplitude_envelope(y, frame_size, hop_size)

frame_scale = np.arange(len(waveform_AE))
# 帧数转换成时间
time_scale = librosa.frames_to_time(frame_scale, sr=sr, hop_length=hop_size)

# 作图原始信号和包络
plt.figure(figsize=(6, 3))
librosa.display.waveshow(y, sr=sr)
plt.plot(time_scale, waveform_AE, c='r')

plt.title("Amplitude Evelope")
plt.show()
```

**示例结果**:

![Figure_1](C:\Users\23606\Pictures\Figure_1.png "结果")





---

## 均方根能量`RMSE`

<img src="C:\Users\23606\Pictures\作业图片\微信截图_20230312161141.png" alt="均方根能量" title="均方根能量" style="zoom: 25%;" />

- 依次寻找每一帧的`RMSE`, 其值为第t帧中每点幅值平方再取均值后开根号

**特点**：均方根能量表达的是一帧内所有样本点的一个综合信息，与时域幅值包络相比，`RMSE`体现了每一帧的包络变化，适用于不平稳的信号，尤其对于突变信号，`RMSE`得到的值较平稳，因为它利用了每一帧所有点幅值的平均值，而不像`AE`只是利用每一帧的最大幅值。

**应用**：音频分割、音频分类



```python
"""
    均方根能量, 主要就是计算函数不同
    手写均方根能量函数cul_rmse
    官方函数为librosa.feature.rms(y, frame_length, hop_length), 
    	返回shape(1, frames)
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def cul_rmse(waveform, frame_length, hop_length):
    """计算均方根能量"""
    if (len(waveform) - frame_length) % hop_length != 0:
        # 这一段实际上是为了确定填充数,使得waveform能被上式整除
        frame_num = int((len(waveform) - frame_length) / hop_length) + 1
        pad_num = frame_num * hop_length + frame_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_num), mode='reflect')
    frame_num = int((len(waveform) - frame_length) / hop_length) + 1        # 帧数

    # 每帧的均方根能量放入列表
    waveform_rms = []
    for i in range(frame_num):
        currect_frame = waveform[i * hop_length: i * hop_length + frame_length]
        currect_rmse = np.sqrt(np.sum(currect_frame ** 2) / frame_length)   # 核心公式
        waveform_rms.append(currect_rmse)

    return np.array(waveform_rms)


file_name = "../audio_data/music_piano.wav"
y, sr = librosa.load(path=file_name, sr=None)
frame_length = 1024
hop_length = int(frame_length / 2)

# shape(877)
waveform_RES = cul_rmse(y, frame_length, hop_length)

frame_scale = np.arange(0, len(waveform_RES))
time_scale = librosa.frames_to_time(frame_scale, sr=sr, hop_length=hop_length)

# 作图波形和均方根能量
fig = plt.figure(figsize=(6, 4), layout='tight')
ax1 = plt.subplot(2, 1, 1)
ax1.plot(time_scale, waveform_RES, color='r')
librosa.display.waveshow(y, sr=sr)
ax1.set_title('Root Mean Square Energy in my')

# waveform_RES_librosa(1, 878), 可以转换一下shape, 该函数会多一帧
waveform_RES_librosa = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0][1:]
ax2 = plt.subplot(2, 1, 2)
ax2.plot(time_scale, waveform_RES, color='r')
librosa.display.waveshow(y, sr=sr)
ax2.set_title("Root Mean Square Energy in librosa")

plt.show()
```

**示例结果**:

<img src="C:\Users\23606\Pictures\Figure_1.png" alt="Figure_1" style="zoom: 80%;" />



