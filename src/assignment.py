import argparse
import logging
import time

import cv2
import numpy as np

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import platform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.CRITICAL)
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

#
POSE_COCO_BODY_PARTS = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
    18: "Background",
}

# call this when a taxi is being hailed!
def hail_taxi(img):
    print("Someone is hailing a taxi!")
    cv2.putText(img, "TAXI!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (94, 218, 255), 2)
    cv2.putText(img, platform.uname().node,
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

if __name__ == '__main__':
    # arguements to your program
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument(
        '--resolution',
        type=str,
        default='432x368',
        help='network input resolution. default=432x368')
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenet_thin',
        help='cmu / mobilenet_thin')
    parser.add_argument(
        '--show-process',
        type=bool,
        default=False,
        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    camera = cv2.VideoCapture(args.camera)
    ret_val, image = camera.read()
    
    print("**** CTRL+C to exit ****")
    while True:
        # get image form the camera
        ret_val, image = camera.read()
        # boilerplate
        canvas = np.zeros_like(image)
        img_scaled = cv2.resize(
            image,
            None,
            fx=args.zoom,
            fy=args.zoom,
            interpolation=cv2.INTER_LINEAR)
        dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
        dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
        canvas[dy:dy + img_scaled.shape[0], dx:
               dx + img_scaled.shape[1]] = img_scaled
        image = canvas
        # feed image into the neural network
        humans = e.inference(image)  # list of humans

        for id, human in enumerate(humans):

            # Create variable to store y coordinate information
            # using nose as it is a singular point 

            nose = 1
            right_wrist = 1
            left_wrist = 1


            # create loop to run through all keys and values in human 
            for key , val in human.body_parts.items():

                # if "Nose" is the current key 
                if POSE_COCO_BODY_PARTS[key] == "Nose":
                    # the variable nose  =  the y cordinate of the Nose. The y value of key Nose  
                    nose = val.y
                    print(nose)

                # if "RWrist" is the current key 
                elif POSE_COCO_BODY_PARTS[key] == "RWrist":
                    # the variable right_wrist  =  the y cordinate of the RWrist. The y value of key RWrist 
                    right_wrist = val.y
                    print(right_wrist)

                # if "LWrist" is the current key 
                elif POSE_COCO_BODY_PARTS[key] == "LWrist":
                    # the variable left_wrist  =  the y cordinate of the LWrist. The y value of key LWrist 
                    left_wrist = val.y 
                    print(left_wrist)

            # Set trigger rules for when a cab is being hailed 
            # The reason for using less than instead of greather than the 0,0 coordinate is the top left of the window 
            if right_wrist < nose or left_wrist < nose:
                hail_taxi(image)



            # my_list = [(POSE_COCO_BODY_PARTS[k], v.x, v.y) for k,v in human.body_parts.items()]
            # print(my_list)

            # RE = 16
            # LE = 17

            # RW = 4
            # LW = 7
            
            # for key, val in human.body_parts.items():
            #     print(human.body_parts[1].y)
            #     print()

            # for k, v in human.body_parts.items():
            #     print(POSE_COCO_BODY_PARTS[k])
            #     print(v.y)
              

            # TODO ensure it only does this when someone is hailing a taxi.
            # That is, an arm is above their head.
            #hail_taxi(image)

            # Debugging statement: remove before demonstration.
            #print([(POSE_COCO_BODY_PARTS[k], v.x, v.y) for k,v in human.body_parts.items()])

            #test co - authour commit 3
            

        # drawing lines on an image
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # FPS counter
        cv2.putText(image, "FPS: {:.2f}".format(1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
