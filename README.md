# Docker_Occlusion

A pose estimator that is more robust to face occlusions. The code includes a gRPC server and client example to perform the estimation in images.


## gRPC Server and Client

### How to experiment
Download this repository. Open two terminals and go to the repository folder directory. On one of the terminals run python OcclusionPose_server.py. On the other run python OcclusionPose_client.py face_image.jpg (face_image will be your image name). It will output the pose and the image with the pose axis.

### Generate protobuf files

If you desire to change the proto file run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. Occlusion_pose.proto
This will generate the pb2 and pb2_grpc files.

## Docker
