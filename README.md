# arlo-false-alarms
Code to detect false alarms in Arlo home camera videos

### Motivation
This code is aimed to detect false alarms when using the Netgear Arlo cameras for personal/home use. I have three of these cameras installed in my home and I've been plagued with false alarms where the camera records video when it thinks it's detected motion but there is actually nothing/no-one moving in the video that is captured. The motion detection on the camera is not done through the camera but through a motion detector - this leaves some scope to process the video and determine if the clip captured was a true/false detection. 

### Basic idea
The basic idea behind this code is to process the video captured by the camera and determine if there is motion within the clip or not. To do this, I wrote some simple code in Python that uses methods implemented in the module OpenCV. The workflow is the following: 
1. Load the video frame by frame
2. Change the frame from RGB to Grayscale 
3. De-noise the gray frame using `cv2.fastNlMeansDenoising`. Details on the parameters are discussed later.
4. Skip the first n frames to avoid sudden changes in contrast when the camera is triggered. This contrast changing effect is most noticeable in bright light, for example when the camera is pointed outdoors
5. Keep the last frame in memory and compute a difference score between the last and current frame using `compare_ssim` from `skimage.measure`. This is a great method that computes the "Structural Similarity Index (SSI)". A higher value indicates more similarity and a lower score indicates more difference. A threshold value for this is chosen based on tests such that an SSIM < threshold is classified as a TRUE detection
6. The de-noising is the bottleneck in the above code. Therefore, ony the first few seconds of the video are considered. The assumption behind this is the following: whatever triggered the motion sensor has to be present in the first few seconds of the video. Therefore there is no need to parse the entire video which can get very time consuming
7. The other idea is to consider a chunk of the video (in 1s intervals) and compute the average SSI across 1s chunks rather than across each frame in the video
8. Finally, only a few seconds (initial) of the video clip are considered to speed-up the computation without a reduction in accuracy
