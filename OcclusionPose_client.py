import logging
import grpc
import numpy as np
import cv2
import utils
import sys

import Occlusion_pose_pb2
import Occlusion_pose_pb2_grpc
import parsing
from matplotlib import pyplot as plt
from PIL import Image

def pose_calc_test(stub,image_path):
    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img = Image.open(image_path)
    #img_msg = parsing.image_to_msg(img)
    with open(image_path, 'rb') as fp:
            image_bytes = fp.read()
    pose_request = Occlusion_pose_pb2.Image(data=image_bytes)
    resp = stub.Process(Occlusion_pose_pb2.PoseRequest(image=image_bytes))
    pose_list = parsing.msg_to_matrix(resp.pose)


    print("Yaw: ",pose_list[0][0]," Pitch: ", pose_list[1][0], " Roll: ", pose_list[2][0])
    utils.draw_axis_pil(img, pose_list[0], pose_list[1], pose_list[2], tdx = img_cv.shape[1] / 2, tdy= img_cv.shape[0] / 2, size = img_cv.shape[1]/2)

    img.show()


    return pose_list, img


if __name__ == '__main__':
    with grpc.insecure_channel('localhost:8061') as channel:
        estimator_stub = Occlusion_pose_pb2_grpc.OcclusionPoseStub(channel)
        try:
            path = sys.argv[1]
            response, img = pose_calc_test(estimator_stub,path)
            #print(response)
            print("Client: Received.")
        except grpc.RpcError as rpc_error:
            print('An error has occurred:')
            print(f'  Error Code: {rpc_error.code()}')
            print(f'  Details: {rpc_error.details()}')
