from CurveFitter import cal_fitted_length, cal_poly_length
import numpy as np
import json
import cv2
import os


def cal_lengths_by_res(left_img, right_img, left_res, right_res):
    H, W, _ = left_img.shape
    left_bboxes = [[r["bbox"]["coord"]["xmin"],
                    r["bbox"]["coord"]["ymin"],
                    r["bbox"]["coord"]["xmax"] - r["bbox"]["coord"]["xmin"],
                    r["bbox"]["coord"]["ymax"] - r["bbox"]["coord"]["ymin"]]
                   for r in left_res]
    right_bboxes = [[r["bbox"]["coord"]["xmin"],
                     r["bbox"]["coord"]["ymin"],
                     r["bbox"]["coord"]["xmax"] - r["bbox"]["coord"]["xmin"],
                     r["bbox"]["coord"]["ymax"] - r["bbox"]["coord"]["ymin"]]
                    for r in right_res]
    # TODO: keypoints can be None!!!
    # left_kpss = [{k: (v["pos"]["x"], v["pos"]["y"]) for k, v in r["keypoints"].items()} for r in left_res]
    # right_kpss = [{k: (v["pos"]["x"], v["pos"]["y"]) for k, v in r["keypoints"].items()} for r in right_res]
    left_kpss = []
    for r in left_res:
        kpss = r["keypoints"]
        kpss_ = {}
        for k, v in kpss.items():
            if v is None:
                kpss_[k] = None
            else:
                kpss_[k] = (v["pos"]["x"], v["pos"]["y"])
        left_kpss.append(kpss_)
    right_kpss = []
    for r in right_res:
        kpss = r["keypoints"]
        kpss_ = {}
        for k, v in kpss.items():
            if v is None:
                kpss_[k] = None
            else:
                kpss_[k] = (v["pos"]["x"], v["pos"]["y"])
        right_kpss.append(kpss_)

    for i, left_bbox in enumerate(left_bboxes):
        # Stereo Bounding Box matching
        right_bbox_matching, i_match = match_stereo_bbox(left_img=left_img, right_img=right_img, left_bbox=left_bbox,
                                                         right_bboxes=right_bboxes,
                                                         vertical_bias=50, similarity_thre=0.2)
        if right_bbox_matching is None:
            continue

        # Word Coordinate Calculation
        left_kps = left_kpss[i]
        right_kps = right_kpss[i_match]
        keypoints_world = cal_keypoints_world_coord(keypointsL=left_kps, keypointsR=right_kps)

        # Parabola Fitting
        can_fit, fit_points = get_fit_points(keypoints_world)
        if not can_fit:
            continue
        curve_length = cal_fitted_length(*fit_points)
        if curve_length is None:
            continue
        left_res[i]["length"] = curve_length
        right_res[i_match]["length"] = curve_length
    return left_res, right_res


def match_stereo_bbox(left_img, right_img, left_bbox, right_bboxes,
                      vertical_bias=10, horizontal_bias=0, similarity_thre=0.8):
    XMIN, YMIN, W, H = map(int, left_bbox)
    left_crop = left_img[YMIN:YMIN+H, XMIN:XMIN+W, :]
    max_similarity = -1
    bbox_matched, matched_id = None, None
    for i, right_bbox in enumerate(right_bboxes):
        xmin, ymin, w, h = map(int, right_bbox)
        if xmin - XMIN > horizontal_bias:
            continue
        if max(abs(YMIN - ymin), abs((YMIN+H) - (ymin+h))) > vertical_bias:
            continue
        right_crop = right_img[ymin:ymin+h, xmin:xmin+w, :]
        right_crop = cv2.resize(right_crop, (left_crop.shape[1], left_crop.shape[0]))
        res = cv2.matchTemplate(left_crop, right_crop, cv2.TM_CCOEFF_NORMED)
        similarity = res[0, 0]
        if similarity >= similarity_thre and similarity > max_similarity:
            max_similarity = similarity
            bbox_matched = right_bbox
            matched_id = i
    return bbox_matched, matched_id


# f = 1, 2.2, 3.6 都他妈试试， 忘记焦距了
# metric: mm
def cal_keypoints_world_coord(keypointsL, keypointsR, B=60, f=1, pix=0.003,
                              keypoint_names=("Head", "Dorsal1", "Dorsal2", "Pectoral", "Gluteal", "Caudal")):
    f = f*1.3
    keypoints_world = {}
    for kname in keypoint_names:
        if keypointsL[kname] is not None and keypointsR[kname] is not None:
            xL, yL = keypointsL[kname][0], keypointsL[kname][1]
            xR, yR = keypointsR[kname][0], keypointsR[kname][1]
            z = (f*B) / ((xL - xR) * pix)
            x = (z/f) * xL * pix
            y = (z/f) * yL * pix
            keypoints_world[kname] = (x, y, z)
        else:
            keypoints_world[kname] = None
    return keypoints_world


def get_fit_points(keypoints):
    Head, DorsalFin1, PectoralFin, DorsalFin2, GlutealFin, CaudalFin = \
        keypoints["Head"], keypoints["Dorsal1"], keypoints["Pectoral"], keypoints["Dorsal2"], keypoints["Gluteal"], keypoints["Caudal"]
    for keypoint in [Head, DorsalFin1, PectoralFin, DorsalFin2, GlutealFin, CaudalFin]:
        if keypoint is None:
            return False, None
    P1 = Head
    P2 = tuple((a + b) / 2 for a, b in zip(DorsalFin1, PectoralFin))
    P3 = tuple((a + b) / 2 for a, b in zip(DorsalFin2, GlutealFin))
    P4 = CaudalFin
    return True, (P1, P2, P3, P4)
