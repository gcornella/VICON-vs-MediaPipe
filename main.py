#!/usr/bin/python

# Import libraries
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import random
import math
import time
from scipy.io import savemat
import datetime
import matplotlib.pyplot as plt
import socket
import input_selection

# Run a simple GUI built using tkinter to choose some inputs
inp, vid = input_selection.calc()
# From binary to boolean
inp = (np.array(inp) > 0).tolist()
vid = (np.array(vid) > 0).tolist()

mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()

## USER INPUTS ##

## FUNCTIONS ##
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

## END FUNCTIONS ##

def run():
    # Input variables from the user
    exercising = inp[0]             # If True, the GUI proposes new knee degrees every time the user pressed the key 'p'
    plot_exercise_line = inp[1]     # If True, the exercise line is going to be plotted on top of the screen
    save = inp[2]                   # If True, the program saves an excel file and a .mat file when it finishes executing
    udp_open = inp[3]               # If True, the UDP communcation is allowed, and then you just have to run the plotter
    if udp_open:
        ip = '127.0.0.1'
        port = 4444
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)

    # If the user selects to import a video, he/she is required to specify the name of the file
    if vid[0]:
        video_file_name = input('Write the name of your video file, and then press the Enter key: ')
        video_file_location = ['videos/' + video_file_name + '.mp4']
    # If the user chooses the webcam option, the computer camera will be activated
    if vid[1]:
        video_file_name = 'webcam_experiment'
        video_file_location = [0]
    # The resolution must be chosen. The biggest resolution selected will be implemented.
    if vid[2]:
        res = (640, 480)
    if vid[3]:
        res = (1280, 720)
    if vid[4]:
        res = (1920, 1080)
    else:
        res = (640, 480)
        print('Select a video format, either webcam or an imported video')
    width, height = res


    t_ini = time.time()     # Start a timer to calculate the total duration of the execution
    cap = cv2.VideoCapture(video_file_location[0]) # Open the camera or the imported video
    if (cap.isOpened() == False):   # Check if camera opened successfully
        print("Error opening video stream or file")

    # Get the number of frames and the fps of the video
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("The total number of frames is {} and the fps {} ".format(frames,fps))
    seconds = round(frames / fps)
    video_time = datetime.timedelta(seconds=seconds)
    print("The total duration of the video is {}".format(video_time))

    # Modifying the resolution if needed
    '''
    reduction_ratio = 0.55
    width = int(cap.get(3) / reduction_ratio)  # float `width`
    height = int(cap.get(4) / reduction_ratio)  # float `height`
    '''

    # Set and define some starting variables for the program
    fr = 0
    next_exercise = False
    global deg
    deg = math.pi / 4
    points = 0

    # Define a df to save the information in .xlsx format
    df = pd.DataFrame(columns=['right_foot_index', 'right_ankle', 'right_knee','right_hip', 'angle_xy', 'angle_xyz','angle_xy_ankle','angle_xyz_ankle','angle_hip'])
    # Define a dictionary to save the information in .mat format
    mat_dict = {'right_ankle':[],'right_foot_index': [],'right_knee': [],'right_hip': [],'angle_xy': [],'angle_xyz':[],'angle_xy_ankle':[],'angle_xyz_ankle':[],'angle_hip':[]}
    final_dict = {'data_python': []}

    ## Setup the mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
        # While the frames are being detected properly, keep executing the video
        while cap.isOpened():
            #print('------ Next frame ------')
            fr += 1 # Increase the variable frame at every iteration
            ret, frame = cap.read()   # Read one frame from the video
            if ret == True:
                # Resize the video according to the chosen resolution
                frame = cv2.resize(frame,(width, height))

                # Flip the video if needed
                frame = cv2.flip(frame, 1)  # Flip the frame vertically
                #frame = cv2.flip(frame, 0) # Flip the frame horizontally

                # # To improve performance, optionally mark the image as not writeable to pass by reference.
                frame.flags.writeable = False

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make the detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract the landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Extract the values for some specific points
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

                    # Set the start point for the exercise line
                    start_x = int(right_knee.x * width)
                    start_y = int(right_knee.y * height)

                    # Sagital plance coordinates (y equals the z coordinate from the Vicon cameras)
                    right_hip_xy = [right_hip.x, right_hip.y]
                    right_knee_xy = [right_knee.x, right_knee.y]
                    right_ankle_xy = [right_ankle.x, right_ankle.y]
                    right_foot_index_xy = [right_foot_index.x, right_foot_index.y]

                    # 3D spaace coordinates
                    right_hip_xyz = (right_hip.x, right_hip.y, right_hip.z)
                    right_knee_xyz = (right_knee.x, right_knee.y, right_knee.z)
                    right_ankle_xyz = (right_ankle.x, right_ankle.y, right_ankle.z)
                    right_foot_index_xyz = (right_foot_index.x, right_foot_index.y, right_foot_index.z)

                    # Angles/Degree calculations of specific joints
                    angle_xy = 180 - calculate_angle(right_hip_xy, right_knee_xy, right_ankle_xy)
                    angle_xyz = 180 - angle_between(tuple(map(lambda i, j: i - j, right_ankle_xyz, right_knee_xyz)),
                                                    tuple(map(lambda i, j: i - j, right_hip_xyz, right_knee_xyz)))*(360/(2*math.pi))
                    angle_xy_ankle = calculate_angle(right_knee_xy, right_ankle_xy, right_foot_index_xy)
                    angle_xyz_ankle = angle_between(tuple(map(lambda i, j: i - j, right_foot_index_xyz, right_ankle_xyz)),
                                                    tuple(map(lambda i, j: i - j, right_knee_xyz, right_ankle_xyz))) * (360 / (2 * math.pi))

                    angle_hip = math.atan2(right_knee.y * height-right_hip.y * height, right_knee.x * width-right_hip.x * width )
                    angle_hip_deg = np.rad2deg(angle_hip)

                    # Print statement for every frame
                    '''
                    print('Knee angle_xy is:  {}'.format(angle_xy))
                    print('Knee angle_xyz is:  {}'.format(angle_xyz))
                    print('Hip angle_xy is:  {}'.format(angle_hip*360 / (2 * math.pi)))
                    print('Deg is:  {}'.format(deg* 360 / (2 * math.pi)))
                    print('Ankle Deg is {}'.format(angle_xyz_ankle))
                    '''

                    if exercising:
                        if next_exercise:
                            deg = round(random.uniform(math.pi/8,math.pi/2),3)
                            next_exercise = False

                    if not exercising:
                        deg = angle_hip

                    dist_knee_ankle = math.sqrt((right_ankle.x* width-right_knee.x* width)**2+(right_knee.y* height-right_ankle.y* height)**2)

                    # Set the end point for the exercise line
                    end_x = int(start_x + dist_knee_ankle * math.cos(deg))
                    end_y = int(start_y + dist_knee_ankle * math.sin(deg))

                    if cv2.waitKey(5) & 0xFF == ord('p'):
                        points += 1
                        next_exercise = True

                    # Plot a line extending the hip-knee to track the exercise until maximum leg stretch
                    if plot_exercise_line:
                        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)

                    # Plot some useful information in another figure to analyze the results
                    if udp_open:
                        MESSAGE = str([int(angle_xy), int(angle_xyz), int(np.rad2deg(deg))]).encode()
                        sock.sendto(MESSAGE, (ip, port))
                        #print("Message sent to plotter: {}".format(MESSAGE))
                        data, address = sock.recvfrom(1024)
                        #print("Message received from plotter: {}".format(data.decode('utf-8')))

                    # Plot a line at a specific angle and try to copy its position
                    '''
                    if not plot_init:
                        # Plot a line
                        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
                        if ((angle_xy < deg * 360 / (2 * math.pi) + 10) and (angle_xy > deg * 360 / (2 * math.pi) - 10)):
                            print('*****************************')
                            plot_init = False
    
                    if plot_init:
                        cv2.line(image, (start_x, start_y), (start_x, start_y + 200), (0, 255, 0), 3)
                        # if ((angle_xy < 100) and (angle_xy > 80)):
                        if cv2.waitKey(33) == 115:
                            plot_init = False
                    '''

                    # Visualize data on top of the video
                    '''
                    ## To put text on top of the screen, the values have to be modified according to the resolution chosen ##
                    align_left = int(width/15)
                    cv2.rectangle(image, (align_left-10,align_left-10), (int(width/3),int(height/2)), (255,255,255), -1)

                    cv2.putText(image, 'Proposed Degree: '+ str(int(deg* 360 / (2 * math.pi))), (align_left, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                    cv2.putText(image, 'Knee Degree: ' + str(int(angle_xyz)), (align_left, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                    cv2.putText(image, 'Hip Degree: ' + str(int(angle_hip*360 / (2 * math.pi))), (align_left, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                    cv2.putText(image, 'Ankle Degree: ' + str(int(angle_xyz_ankle)), (align_left, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                    cv2.putText(image, 'Timer: ' + str(int(time.time()-t_ini)), (align_left, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(36, 255, 12), 2)
                    cv2.putText(image, 'Points: ' + str(int(points)), (align_left, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                    '''
                    if save:
                        # Save data to dataframe, readable by Excel as a .xlsx file
                        df.loc[fr, ['right_foot_index']] = right_foot_index
                        df.loc[fr, ['right_ankle']] = right_ankle
                        df.loc[fr, ['right_knee']] = right_knee
                        df.loc[fr, ['right_hip']] = right_hip
                        df.loc[fr, ['angle_xy']] = angle_xy
                        df.loc[fr, ['angle_xyz']] = angle_xyz
                        df.loc[fr, ['angle_xy_ankle']] = angle_xy_ankle
                        df.loc[fr, ['angle_xyz_ankle']] = angle_xyz_ankle
                        df.loc[fr, ['angle_hip']] = angle_hip_deg

                        # Save data to dictionary, readable by Matlab as a .mat file
                        mat_dict['right_foot_index'].append([right_foot_index.x, right_foot_index.z, right_foot_index.y])
                        mat_dict['right_ankle'].append([right_ankle.x, right_ankle.z, right_ankle.y])
                        mat_dict['right_knee'].append([right_knee.x, right_knee.z, right_knee.y])
                        mat_dict['right_hip'].append([right_hip.x, right_hip.z, right_hip.y])
                        mat_dict['angle_xy'].append([angle_xy])
                        mat_dict['angle_xyz'].append([angle_xyz])
                        mat_dict['angle_xy_ankle'].append([angle_xy_ankle])
                        mat_dict['angle_xyz_ankle'].append([angle_xyz_ankle])
                        mat_dict['angle_hip'].append([angle_hip_deg])
                        final_dict = {'data_python': mat_dict}
                except:
                    pass

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                cv2.imshow('Mediapipe Feed', image)

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
                print("Can't receive frame (stream end?). Exiting ...")

        cap.release()
        cv2.destroyAllWindows()
        print("The total duration of the analyzed video is {}".format(time.time()-t_ini))

    # Save info for external use
    if save:
        df.to_excel('data/'+video_file_name+".xlsx", sheet_name=video_file_name)
        savemat('data/'+video_file_name+".mat", final_dict)

    if udp_open:
        print('Closing socket')
        MESSAGE = str([1, 100]).encode()
        sock.sendto(MESSAGE, (ip, port))
        sock.close()

    # Press the green button in the gutter to run the script.

if __name__ == '__main__':
    run()
