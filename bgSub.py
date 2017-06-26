import cv2
import os
from skimage.measure import compare_ssim
import time
import json
import sys
import argparse

"""This function analyzes a video and returns a 3-tuple 
   Inputs:
   video:           The name of the video to be analyzed 
   path:            The path where the video can be found (locally)
   skip_frames:     Number of frames to skip at the beginning of the video
   fps:             Framerate of the video to be analyzed
   secs_consider:   Number of seconds to consider for analyzing the video
   thres:           Threshold value (0-1) for the SSI which determines if this 
                    will be flagged as a false alarm or not
   debug:           Flag for turning on/off debug information

   Outputs:
   (val1, val2, val3)
   val1:            True/False: True if there is motion detected, False otherwise
   val2:            The average SSI score for the frames considered
   val3:            The elapsed time to flag the video as True/False
"""
def analyze_video(video, path, skip_frames=12, fps=24, secs_consider=3, thres=0.997, debug=False): 
    # Make sure this is a video (.mp4) format
    if ( len(video.split(".")) > 1 and video.split(".")[1] == "mp4" ): 
        
        print("Analyzing {}...".format(video))
        
        # Complete path to locate video
        vid_path = os.path.join(path,video)

        # Start the video "reader"
        cap = cv2.VideoCapture(vid_path)

        # Get number of frames in the video (CAP_PROP_FRAME_COUNT is now 7?!)
        # num_frames = int(cap.get(cv2.CV_CAP_PROP_FRAME_COUNT))
        num_frames = int(cap.get(7))
        if ( debug ):
            print("\tNum frames:\t\t{}".format(num_frames)) 
            
        if ( num_frames==0 ):
            # Empty video or cv2 does not like this video
            print("The video {} is empty! Skipping...".format(video))
            return (-1,-1,-1)
        
        # Determine the start index for frames to analyze. One second worth of video
        # is analyzed starting from the frame start indices
        # This is basically 
        #       one second from the start (after skipping frames, if specified)
        #       one second in the middle
        #       one second from the end
        frame_start_analyze = [skip_frames + 1, num_frames//2, num_frames - fps - 1]
        if ( debug ):
            print("\tFrame pos :\t\t{}".format(frame_start_analyze))

        # Check if the video is long enough
        if ( ( frame_start_analyze[1] < frame_start_analyze[1] ) or\
             ( frame_start_analyze[2] < frame_start_analyze[1] ) ):
             # Not long enough
             print("The video [{}]is not long enough. Try omitting".format(video)) 
             print("\"skip_frames\" or recording longer video. Skipping... ")
             return (-1,-1,-1)
        
        # Setup some parameters
        ssim_scores_dn = []
        sliding_frame_score = []
        motion_detected = False
        frames_per_section = int ( fps * secs_consider / 3 )

        # Start the timer
        start = time.time()
        
        # Loop through the starting indices for the frames
        for start_idx in frame_start_analyze:
            
            # Set the marker at the start_idx (CAP_PROP_POS_FRAMES is now 1?!)
            # cap.set(cv2.CV_CAP_PROP_POS_FRAMES, start_idx)
            cap.set(1, start_idx)

            # Some parameters
            frame_count = 0
            first_frame = False

            # Loop through the frames
            while(1):
                ret, frame = cap.read()
                if ( ret and frame_count <= frames_per_section ):

                    # Convert frame to grayscale and de-noise it
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    denoised_gray_frame = cv2.fastNlMeansDenoising(gray_frame,None,20,7,11)

                    # If this is the first frame, do nothing (nothing to compare against)
                    if ( first_frame == False ):
                        old_denoised_frame = denoised_gray_frame
                        first_frame = True
                    
                    # Otherwise, calculate SSI score and append it to a list
                    else:
                        score_dn, diff_dn = compare_ssim(old_denoised_frame, denoised_gray_frame, full=True)
                        sliding_frame_score.append(score_dn)
                        old_denoised_frame = denoised_gray_frame
                
                # Already traversed through the required number of frames?
                elif ( frame_count > frames_per_section ):
                    ssim_scores_dn.append(sum(sliding_frame_score)/len(sliding_frame_score))
                    if (debug):
                        print("\tScore fr  :\t\t{}".format(ssim_scores_dn[-1]))
                    sliding_frame_score = []
                    break
                
                # For some reason, the next frame could not be read
                elif ( ret == False ):
                    print("Error reading frame number {0} in video {1}!".format( start_idx + frame_count, video ) )
                    break
                
                # Increment frame count
                frame_count += 1

            # If SSI average score for the last sliding window was less than threshold
            # Then motion was detected. Break here. No need to parse the rest of the video                             
            if ( ssim_scores_dn[-1] < thres ):
                if (debug):
                    print("\tMotion detected! : {}".format(ssim_scores_dn[-1]))
                motion_detected = True
                break
        
        # Stop the timer
        elapsed_time = time.time() - start

        # Get average SSI score from the three sliding frame windows
        ssim_avg_dn = sum(ssim_scores_dn)/len(ssim_scores_dn)

        # Judge. 
        if ( ssim_avg_dn > thres ):
            if (debug):
                print("{0} is a FALSE detection; avg score: {1}, time spent: {2}".format(video, ssim_avg_dn, elapsed_time))
            return (False, ssim_avg_dn, elapsed_time)

        else:
            if (debug):
                print("{0} is a TRUE detection; avg score: {1}, time spent: {2}".format(video, ssim_avg_dn, elapsed_time))
            return (True, ssim_avg_dn, elapsed_time)

        # Release the video file
        cap.release()
    
    else:
        print("Check the video file [{}]. This is not an MP4!".format(video))
        return (-1,-1,-1)


"""Simple function to parse bool from string"""
def true_false(label):
    to_upper = str(label).upper()
    if ( to_upper == "TRUE" ):
        return True
    elif ( to_upper == "FALSE" ):
        return False
    else:
        print("Incorrect argument specified for --lab. Please choose between")
        print("True or False.")
        exit()


"""This function loads the result JSON file and displays a 
   summary."""
def displaySummary(json_file, label):
    with open(json_file) as data_file:
        data = json.load(data_file)
    
    total_vids = len ( data )
    print("\n")
    print("***************** Video Analysis Summary *****************")
    print("\n")
    print("Videos Analyzed :\t\t{}\n".format( total_vids ) )
    print("Video Name                 Motion Detected?   Time Elapsed (s)")
    print("---------------------------------------------------------------")
        
    for key in data:
        res = data[key]
        if ( res[1] == True ):
            print("{0: <26} {1}                {2}".format(res[0], res[1], round(res[3],2)))
        elif ( res[1] == False ):
            print("{0: <26} {1}               {2}".format(res[0], res[1], round(res[3],2)))
        else:
            print("{0: <26} Error               Error".format(res[0]))

    if ( label is not None ):
        incorrect_classification = []
        correct_classification = []
        true_true = 0
        true_false = 0
        
        for key in data:
            res = data[key]
            if ( res[1] != label ):
                incorrect_classification.append([res[0], label])
                if ( label ):
                    true_false += 1
            else:
                correct_classification.append([res[0], res[1]])
                if ( label ):
                    true_true += 1
        
        false_true = len(incorrect_classification) - true_false
        false_false = len(correct_classification) - true_true
        
        print()
        if ( true_true + true_false ):
            print("True detections :\t\t{0}\{1}".format( true_true, true_false + true_true ))
        if ( false_false + false_true ):
            print("False detections:\t\t{0}\{1}".format( false_false, false_false + false_true ))
        print("Total success rate :\t\t{}%".format( 100*len(correct_classification)/total_vids ))
        print()
        print("Incorrectly classified videos:")
        for item in incorrect_classification:
            print("{0} \twas\t{1}\tclassified as {2}".format( item[0], item[1], not item[1] ))

def main():

    parser = argparse.ArgumentParser(description='Analyze Arlo video for motion detection')
    parser.add_argument("path", help="Path to the video(s) you want analyzed")
    parser.add_argument("--f", help="Frame per second of the video (integer). Default is 24", type=int)
    parser.add_argument("--s", help="Number of frames to skip from the start (integer). Default is 12", type=int)
    parser.add_argument("--l", "--length", help="Length of the video(s) in seconds to analyze (real). Default is 3s", type=float)
    parser.add_argument("result", help="Name and path of result (.json) file")
    parser.add_argument("--t", "--threshold", help="SSI below this number indicates motion (real). Default is 0.997", type=float)
    parser.add_argument("--v", help="Verbose output? True/False. Default is False", type=bool)
    parser.add_argument("--lab", help="Known label for videos in path. True/False", type=true_false)
    args = parser.parse_args()
    
    path = args.path 
    videos = os.listdir(path)
    result = {}
    outfile = args.result
    
    # Default values for optional parameters
    if ( args.f ):
        fps = args.f
    else:
        fps = 24
    
    if ( args.s ):
        skip_frames = args.s
    else:
        skip_frames = 12
    
    if ( args.l ):
        length = args.l 
    else:
        length = 3

    out_file = args.result
    
    if ( args.v is not None ):
        verbose = args.v
    else:
        verbose = False
    
    if ( args.t ):
        threshold = args.t
    else:
        threshold = 0.997
    
    if ( args.lab is not None ):
        label = bool(args.lab)
    else:
        label = None
    
    # Run through all videos in the path
    for idx,video in enumerate(videos):
        result[video] = (video, ) + analyze_video(video,path,\
                                                         skip_frames=skip_frames,\
                                                         fps = fps,\
                                                         secs_consider = length,\
                                                         thres = threshold,\
                                                         debug=verbose)

    # Write results to output file
    with open(out_file,"w") as f:
        json.dump(result,f)

    # Display summary report
    displaySummary(outfile, label) #"../data/videos/result_mid_frames.json")

if __name__ == "__main__":
    main()