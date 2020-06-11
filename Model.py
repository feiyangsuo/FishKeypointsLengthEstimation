from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor


def get_keypoint_detection_model(num_classes=2, num_keypoints=6, device=None):
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # 本来就只有两类
    in_features_keypoint = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_keypoint, num_keypoints)
    model.to(device)
    return model


if __name__ == '__main__':
    model = get_keypoint_detection_model()
    pass
