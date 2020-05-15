"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_thresh", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def model_out(frame, result):
    
    current_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_thresh:
            xmin = int(obj[3] * initial_width)
            ymin = int(obj[4] * initial_height)
            xmax = int(obj[5] * initial_width)
            ymax = int(obj[6] * initial_height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count

def connect_mqtt():
    
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

    
def main():

    args = build_argparser().parse_args()
     
    client = connect_mqtt()

    global initial_width, initial_height,prob_thresh

        # Initialise the class
    network = Network()
    # Set Probability threshold for detections
    if args.prob_thresh is None:
        prob_thresh= args.prob_thresh
    else:
        prob_thresh=0.4
            

    image_mode = False

    cur_request_id = 0
    end = 0
    total = 0
    start = 0

    # Load the network to IE plugin to get shape of input layer
    
    n, c, h, w = network.load_model(args.model, args.device, 1, 1,
                                          cur_request_id, args.cpu_extension)[1]

    
    if args.input == 'CAM':
        inp_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        image_mode = True
        inp_stream = args.input

    # Checks for video file
    else:
        inp_stream = args.input
       
    capture = cv2.VideoCapture(inp_stream)

    if inp_stream:
        capture.open(args.input)

    
    initial_width = capture.get(3)
    initial_height = capture.get(4)
    
    while capture.isOpened():
        flag, frame = capture.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Start async inference
 
        image = cv2.resize(frame, (w, h))
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        # Start asynchronous inference for specified request.
        
        inference_start = time.time()
        network.exec_net(cur_request_id, image)
        # Wait for the result
        
        if network.wait(cur_request_id) == 0:
            det_time = time.time() - inference_start
            # Results of the output layer of the network
        
        result = network.get_output(cur_request_id)
        frame, current_count = model_out(frame, result)
            
        inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            
        cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            # When new person enters the video
            
        if current_count > end:
            start = time.time()
            total+= current_count - end
            client.publish("person", json.dumps({"total": total}))

            # Person duration in the video is calculated
        if current_count < end:
            duration = int(time.time() - start)
               # Publish messages to the MQTT server
            client.publish("person/duration",
                              json.dumps({"duration": duration}))

       
        client.publish("person", json.dumps({"count": current_count}))
        end = current_count    
    

        if key_pressed == 27:
                break

        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        if image_mode:
            cv2.imwrite('output_image.jpg', frame)
    
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()
    network.clean()

if __name__ == '__main__':
    main()
