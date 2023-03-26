# 使用arecord、aplay 实现录音和播放



## arecord录音



1.   **查看录音设备**

	- `arecord -l` 	 

		

2. **录音**

	- `arecord -Dhw:0,1 -d 10 -r 44100 -c 1  -f cd -t wav test.wav`

		

- 参数解析
	-D 指定了录音设备，0,1 是card 0 device 1的意思
	-d 指定录音的时长，单位时秒
	-r  指定了采样率，单位时Hz
	-c  指定channel 个数

	-f  指定录音格式，通过上面的信息知道只支持 cd cdr dat 

	-t  指定生成的文件格式



## aplay播放

  **查看播放设备**

- `aplay -l` 	 



**播放音频**

- `aplay test.wav`



