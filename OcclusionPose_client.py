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

# def get_request_homog_calc(matrix1, matrix2):
#     """Returns a message of type HomogCalcRequest.
#     Keyword arguments:
#     matrix1 -- source points to append to message
#     matrix2 -- destination points to append to message
#     """
#     matrix1_msg = parsing.matrix_to_msg(matrix1)
#     matrix2_msg = parsing.matrix_to_msg(matrix2)
#     return vision_algorithms_pb2.HomogCalcRequest(source_pts=matrix1_msg, dest_pts=matrix2_msg, ransac_thresh=5,
#                                                   max_ransac_iters=200)


# def homog_calc_test(stub):
#     # Generate ground truth homography
#     angle = 0.1
#     rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
#                          [np.sin(angle), np.cos(angle), 0],
#                          [0, 0, 1]])
#     a = 0
#     b = 0
#     affine = np.array([[1, a, 0],
#                        [b, 1, 0],
#                        [0, 0, 1]])
#     c = 0
#     projective = np.array([[1, 0, 0],
#                            [0, 1, 0],
#                            [c, c, 1]])
#     true_homog = rotation @ affine @ projective

#     # Generate source and destination points, according to ground truth homography
#     src_pts = np.random.rand(2, 5) * 100
#     dst_pts = true_homog @ np.r_[src_pts, np.ones((1, src_pts.shape[1]))]
#     dst_pts[0, :] = dst_pts[0, :] / dst_pts[2, :]
#     dst_pts[1, :] = dst_pts[1, :] / dst_pts[2, :]
#     dst_pts = np.delete(dst_pts, 2, 0)

#     homog_request = get_request_homog_calc(src_pts.T, dst_pts.T)
#     return stub.Process(vision_algorithms_pb2.ExecRequest(homog_calc_args=homog_request))


# def homog_warp_test(stub, homog_msg):
#     img = np.random.random_sample((3, 300, 600)) * 255
#     img_msg = parsing.image_to_msg(img)
#     warp_request = vision_algorithms_pb2.HomogWarpRequest(image=img_msg, homography=homog_msg, out_width=10,
#                                                           out_height=10)
#     return stub.Process(vision_algorithms_pb2.ExecRequest(homog_warp_args=warp_request))


# def sift_det_test(stub):
#     img = cv2.imread('car.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_msg = parsing.image_to_msg(img)

#     sift_request = vision_algorithms_pb2.SiftDetRequest(image=img_msg)
#     resp = stub.Process(vision_algorithms_pb2.ExecRequest(sift_det_args=sift_request))
#     # keypts = parsing.msg_to_matrix(resp.sift_det_out.keypoints)
#     #
#     # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # img = cv2.drawKeypoints(gray_img, keypts, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     # cv2.imshow(img)
#     # if cv2.waitKey(1000) & 0xFF == ord('q'):
#     #     cv2.destroyAllWindows()

#     return resp

def pose_calc_test(stub,image_path):
    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img = Image.open(image_path)
    #img_msg = parsing.image_to_msg(img)
    with open(image_path, 'rb') as fp:
            image_bytes = fp.read()
    pose_request = Occlusion_pose_pb2.Image(data=image_bytes)
    #pose_request = Occlusion_pose_pb2.PoseRequest(image=img_msg)
    #resp = stub.Process(Occlusion_pose_pb2.PoseRequest(image= img_msg))
    resp = stub.Process(Occlusion_pose_pb2.PoseRequest(image=image_bytes))
    #pose_list = parsing.msg_to_matrix(resp.pose_out.pose)
    pose_list = parsing.msg_to_matrix(resp.pose)
    #img = cv2.drawKeypoints(gray_img, keypts, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #print(pose_list[0])
    #cv2.startWindowThread()
    #cv2.namedWindow("preview")
    #plt.imshow(img)
    #plt.show()
    #cv2.imshow('preview',img)
    #if cv2.waitKey(1000) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
    #print(pose_list)

    print("Yaw: ",pose_list[0][0]," Pitch: ", pose_list[1][0], " Roll: ", pose_list[2][0])
    #utils.draw_axis(img, pose_list[0], pose_list[1], pose_list[2], tdx = img.shape[1] / 2, tdy= img.shape[0] / 2, size = img.shape[1]/2)
    utils.draw_axis_pil(img, pose_list[0], pose_list[1], pose_list[2], tdx = img_cv.shape[1] / 2, tdy= img_cv.shape[0] / 2, size = img_cv.shape[1]/2)
    #plt.imshow(img)
    #plt.show(block=False)
    #plt.show()
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img.show()
    #img.save('a.jpg')
    #cv2.imshow('',img)
    #cv2.waitKey(0)

    return pose_list, img


if __name__ == '__main__':
    # args = parse_args()
    #with grpc.insecure_channel('localhost:50051') as channel:
    with grpc.insecure_channel('localhost:8061') as channel:
        estimator_stub = Occlusion_pose_pb2_grpc.OcclusionPoseStub(channel)
        try:
            # response = homog_calc_test(estimator_stub)
            # print("Client: Received homography ", parsing.msg_to_matrix(response.homog_calc_out.homography))

            # response = homog_warp_test(estimator_stub, response.homog_calc_out.homography)
            # print("Client: Received warped image.")
            #path = 'face_test.jpg'
            path = sys.argv[1]
            response, img = pose_calc_test(estimator_stub,path)
            #print(response)
            print("Client: Received.")
        except grpc.RpcError as rpc_error:
            print('An error has occurred:')
            print(f'  Error Code: {rpc_error.code()}')
            print(f'  Details: {rpc_error.details()}')