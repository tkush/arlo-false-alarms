# arlo-false-alarms
Code to detect false alarms in Arlo home camera videos

### Motivation/Use case
This code is aimed to detect false alarms when using the Netgear Arlo cameras for personal/home use. I have three of these cameras installed in my home and I've been plagued with false alarms where the camera records video when it thinks it's detected motion but there is actually nothing/no-one moving in the video that is captured. The motion detection on the camera is not done through the camera but through a motion detector - this leaves some scope to process the video and determine if the clip captured was a true/false detection. 
Broadly, there are two use cases that such a processing of the video can be applied to:
1. The first is to do an **online** processing. This means that as soon as the camera detects motion and records the video, this program should process that video and determine whether it was a false alarm or not. Automation such as an email message/app notification would suit this use case very well. This is how Arlo currently functions - there is a notification on my phone almost as soon as "motion" is detected. 
2. The second is to do a **batch-processing** of all the videos recorded through the day/week/etc. This is done by manually downloading all the recorded videos from the Arlo website and then running them through the code presented here - with an output of "True" or "False" for the motion detection. **The code presented here is catered to this use case**.

### Basic idea
The basic idea behind this code is to process the video captured by the camera and determine if there is motion within the clip or not. To do this, I wrote some simple code in Python that uses methods implemented in the module OpenCV. The workflow is the following: 
1. Load the video frame by frame
2. Change the frame from RGB to Grayscale 
3. De-noise the gray frame using `cv2.fastNlMeansDenoising`. Details on the parameters are discussed later.
4. Skip the first n frames to avoid sudden changes in contrast when the camera is triggered. This contrast changing effect is most noticeable in bright light, for example when the camera is pointed outdoors. Sometimes the contrast variation changes pixed values greatly and that will lead to incorrect results
5. Keep the last frame in memory and compute a difference score between the last and current frame using `compare_ssim` from `skimage.measure`. This is a great method that computes the "Structural Similarity Index (SSI)". A higher value indicates more similarity and a lower score indicates more difference. A threshold value for this is chosen based on tests such that an SSIM < threshold is classified as a TRUE detection
6. The de-noising is the bottleneck in the above code. Therefore, ony few seconds of the video are considered. The frames to be considered are divided between the start, mid and end of the video for higher detection robustness. The assumption behind this is the following: 
 * Whatever triggered the motion sensor has to be present in the first few seconds of the video. 
 * If the object in the video is stationary at the start, then this pipeline will fail. That is the motivation for looking at the middle and end of the video as well. 
 Therefore there is no need to parse the entire video which can get very time consuming

### Usage
The entire code is just one Python file - `bgSub.py`
```
python bgSub.py -h
usage: bgSub.py [-h] [--f F] [--s S] [--l L] [--t T] [--v V] [--lab LAB]
                path result

Analyze Arlo video for motion detection

positional arguments:
  path                  Path to the video(s) you want analyzed
  result                Name and path of result (.json) file

optional arguments:
  -h, --help            show this help message and exit
  --f F                 Frame per second of the video (integer). Default is 24
  --s S                 Number of frames to skip from the start (integer).
                        Default is 12
  --l L, --length L     Length of the video(s) in seconds to analyze (real).
                        Default is 3s
  --t T, --threshold T  SSI below this number indicates motion (real). Default
                        is 0.997
  --v V                 Verbose output? True/False. Default is False
  --lab LAB             Known label for videos in path. True/False
```
For testing the code on known videos (videos known to be TRUE or FALSE detections beforehand), supply the flag `--lab TRUE/FALSE`.
A 4-tuple is written to the JSON output file which is `(name_of_video, motion_detected_true_or_false, average_SSI_score, time_elapsed_in_analysis)`.

### Results
This code was tested on video captured from my home's Arlo camera on 178 videos of length varying from 3s to 10s (the standard length of video captured on my cameras). These videos were pre-labelled as True (152) or False (26) videos. This code achieves an accuracy of **92.69%** (139/152 correct true detections and 26/26 correct false detections) with the following parameters: 
* FPS set to 24
* First 12 frames skipped
* 3 seconds of video considered (1s at the start, 1s in the middle and 1s from the end)
* Threshold SSI of 99.7%
Here are some examples of True/False detections:

<img src="/images/TRUE.gif" alt="Video correctly classified as True detection" width="320" height="176" />

Correct classification (motion detected)

<img src="/images/FALSE.gif" alt="Video correctly classified as False detection" width="320" height="176" />

Correct classification (no motion detected)

The approach presented here is advantageous in that it can be applied to a different camera feed without any modifications to the code since the basic idea remains the same irrespective of where the camera is pointed or what the lighting conditions are. 

Feel free to post comments/fork this repo or just drop me a line! 

### Pre-requisites to run
This code was developed and tested using the following: 
* Python 3.5.2
* OpenCV 3.1.0
* SciKit Image 0.12.3
You're probably best using the same or higher versions :)

### TO-DO
- [x] Upload the final code to repo
- [x] Convert code to accept command line arguments
- [x] Insert more information in README (pictures, explanation of denoising, timing)
- [ ] Add code for getting video from the Arlo website
- [ ] Host code on a simple server and make the workflow automatic
- [ ] OR plug this into IFTTT
- [ ] Add code to detect objects in the TRUE detections (object recognition using RCNN?)
