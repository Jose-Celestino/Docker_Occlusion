# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import Occlusion_pose_pb2 as Occlusion__pose__pb2


class OcclusionPoseStub(object):
    """Service that receives a request, executes requested algorithm on that image
    and returns the corresponding output
    :param ExecRequest: The request specifying the algorithm and its inputs
    :returns: The corresponding output from the algorithm
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Process = channel.unary_unary(
                '/OcclusionPose/Process',
                request_serializer=Occlusion__pose__pb2.Image.SerializeToString,
                response_deserializer=Occlusion__pose__pb2.PoseResponse.FromString,
                )


class OcclusionPoseServicer(object):
    """Service that receives a request, executes requested algorithm on that image
    and returns the corresponding output
    :param ExecRequest: The request specifying the algorithm and its inputs
    :returns: The corresponding output from the algorithm
    """

    def Process(self, request, context):
        """rpc Process(PoseRequest) returns (PoseResponse);
        rpc Process(Image) returns (Image);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_OcclusionPoseServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Process': grpc.unary_unary_rpc_method_handler(
                    servicer.Process,
                    request_deserializer=Occlusion__pose__pb2.Image.FromString,
                    response_serializer=Occlusion__pose__pb2.PoseResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'OcclusionPose', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class OcclusionPose(object):
    """Service that receives a request, executes requested algorithm on that image
    and returns the corresponding output
    :param ExecRequest: The request specifying the algorithm and its inputs
    :returns: The corresponding output from the algorithm
    """

    @staticmethod
    def Process(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/OcclusionPose/Process',
            Occlusion__pose__pb2.Image.SerializeToString,
            Occlusion__pose__pb2.PoseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
