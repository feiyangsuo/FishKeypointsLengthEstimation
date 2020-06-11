import torch
from DataLoader import get_dataloader, get_transforms
from Model import get_keypoint_detection_model
from StereoLength import cal_lengths_by_res
from tqdm import tqdm
import os
import cv2
import json
import time


class WarmUpLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warm_up_epochs, warm_up_factor):
        def f(epoch):
            if epoch >= warm_up_epochs:
                return 1
            alpha = epoch / warm_up_epochs
            return warm_up_factor * (1-alpha) + 1.0 * alpha
        super(WarmUpLR, self).__init__(optimizer, f)


def train_and_eval(epochs, batch_size, batches_show=10, val_split=0.0,
                   save_dir="work_dir"):
    os.makedirs(save_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is unavailable, using CPU instead.")

    dataloader_train, dataloader_val = get_dataloader(train=True, batch_size=batch_size, val_split=val_split, shuffle=True, num_workers=0)
    val = False if dataloader_val is None else True
    if val:
        raise Exception("I don't want to implement this. So don't validate.")

    model = get_keypoint_detection_model(num_classes=2, num_keypoints=6, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.train()
    for epoch in range(epochs):
        print("----------------------  TRAINING  ---------------------- ")
        running_loss = 0.0
        lr_scheduler_warmup = None
        if epoch == 0:
            lr_scheduler_warmup = WarmUpLR(optimizer, warm_up_epochs=min(1000, len(dataloader_train)),
                                           warm_up_factor=0.001)
            # Only warm up at the first epoch. In this case, lr_scheduler steps at not each epoch but each batch.
        for i, data in enumerate(dataloader_train):
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum([loss for loss in loss_dict.values()])
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if lr_scheduler_warmup is not None:
                lr_scheduler_warmup.step()  # This steps at each batch

            running_loss += losses.item()
            if (i + 1) % batches_show == 0:
                print('[epoch: {}, batch: {}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / batches_show))
                running_loss = 0.0
        lr_scheduler.step()  # This steps at each epoch

        if val:
            raise Exception("I don't want to implement this. So don't validate.")

        print("----------------------   SAVING   ---------------------- ")
        torch.save(model, os.path.join(save_dir, "epoch_{}.pth".format(epoch)))
        torch.save(model.state_dict(), os.path.join(save_dir, "epoch_{}.state_dict.pth".format(epoch)))


def test_model(model_path, candidates_dir, is_state_dict=False,
               kp_colors=((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)),
               kp_names=("Head", "Dorsal1", "Dorsal2", "Pectoral", "Gluteal", "Caudal"),
               box_score_thre=0.5, kp_score_thre=0,
               save_image_out=True, save_json_out=True):
    out_dir = candidates_dir + ".detected"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if is_state_dict:
        model = get_keypoint_detection_model(num_classes=2, num_keypoints=6, device=device)
        model.load_state_dict(torch.load(model_path))
    else:
        model = torch.load(model_path).to(device)
    model.eval()

    json_out = {}
    transforms = get_transforms(train=False)
    for p in tqdm(os.listdir(candidates_dir)):
        image_path = os.path.join(candidates_dir, p)
        image_cv2 = cv2.imread(image_path)
        b, g, r = cv2.split(image_cv2)
        image = cv2.merge([r, g, b])
        image, _ = transforms(image, None)
        image = [image.to(device)]
        pred = model(image)[0]

        scores = pred["scores"]
        boxes = pred["boxes"]
        keypointss = pred["keypoints"]
        keypoints_scoress = pred["keypoints_scores"]

        if save_image_out:
            for i in range(scores.shape[0]):
                if scores[i] >= box_score_thre:
                    box = boxes[i]
                    keypoints = keypointss[i]
                    keypoints_scores = keypoints_scoress[i]
                    cv2.rectangle(image_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                  color=(0, 255, 255), thickness=2)
                    for j in range(keypoints_scores.shape[0]):
                        kp = keypoints[j]
                        if kp[2] >= 0.5 and keypoints_scores[j] >= kp_score_thre:
                            cv2.circle(image_cv2, (int(kp[0]), int(kp[1])), color=kp_colors[j], thickness=2, radius=2)
            cv2.imwrite(os.path.join(out_dir, p), image_cv2)

        if save_json_out:
            json_out[p] = []
            for i in range(scores.shape[0]):
                if scores[i] >= box_score_thre:
                    obj = {"bbox": None, "keypoints": None}
                    box = boxes[i]
                    keypoints = keypointss[i]
                    keypoints_scores = keypoints_scoress[i]
                    obj["bbox"] = {"coord": {"xmin": float(box[0]),
                                             "ymin": float(box[1]),
                                             "xmax": float(box[2]),
                                             "ymax": float(box[3])},
                                   "score": float(scores[i])}
                    json_keypoints = {k: None for k in kp_names}
                    for j in range(keypoints_scores.shape[0]):
                        kp = keypoints[j]
                        if kp[2] >= 0.5 and keypoints_scores[j] >= kp_score_thre:
                            json_keypoints[kp_names[j]] = {"pos": {"x": float(kp[0]),
                                                                   "y": float(kp[1])},
                                                           "score": float(keypoints_scores[j])}
                    obj["keypoints"] = json_keypoints
                    json_out[p].append(obj)
    if save_json_out:
        with open(os.path.join(candidates_dir, "..", "result.json"), "w") as f:
            json.dump(json_out, f)


# ################################# Usage ################################## #

def load_model_eval(model_path, is_state_dict, use_cpu=False):
    device = torch.device("cuda") if (torch.cuda.is_available() and not use_cpu) else torch.device("cpu")
    if is_state_dict:
        model = get_keypoint_detection_model(num_classes=2, num_keypoints=6, device=device)
        model.load_state_dict(torch.load(model_path))
    else:
        model = torch.load(model_path).to(device)
    model.eval()
    transforms = get_transforms(train=False)
    return model, transforms, device


def predict(image_cv2, model, transforms, device,
            kp_colors=((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)),
            kp_names=("Head", "Dorsal1", "Dorsal2", "Pectoral", "Gluteal", "Caudal"),
            box_score_thre=0.5, kp_score_thre=0,
            show=False):
    b, g, r = cv2.split(image_cv2)
    image = cv2.merge([r, g, b])
    image, _ = transforms(image, None)
    image = [image.to(device)]
    pred = model(image)[0]

    scores = pred["scores"]
    boxes = pred["boxes"]
    keypointss = pred["keypoints"]
    keypoints_scoress = pred["keypoints_scores"]

    if show:
        for i in range(scores.shape[0]):
            if scores[i] >= box_score_thre:
                box = boxes[i]
                keypoints = keypointss[i]
                keypoints_scores = keypoints_scoress[i]
                cv2.rectangle(image_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                              color=(0, 255, 255), thickness=2)
                for j in range(keypoints_scores.shape[0]):
                    kp = keypoints[j]
                    if kp[2] >= 0.5 and keypoints_scores[j] >= kp_score_thre:
                        cv2.circle(image_cv2, (int(kp[0]), int(kp[1])), color=kp_colors[j], thickness=2, radius=2)
        cv2.imshow("Result", image_cv2)
        cv2.waitKey(0)

    res = []
    for i in range(scores.shape[0]):
        if scores[i] >= box_score_thre:
            obj = {"bbox": None, "keypoints": None}
            box = boxes[i]
            keypoints = keypointss[i]
            keypoints_scores = keypoints_scoress[i]
            obj["bbox"] = {"coord": {"xmin": float(box[0]),
                                     "ymin": float(box[1]),
                                     "xmax": float(box[2]),
                                     "ymax": float(box[3])},
                           "score": float(scores[i])}
            json_keypoints = {k: None for k in kp_names}
            for j in range(keypoints_scores.shape[0]):
                kp = keypoints[j]
                if kp[2] >= 0.5 and keypoints_scores[j] >= kp_score_thre:
                    json_keypoints[kp_names[j]] = {"pos": {"x": float(kp[0]),
                                                           "y": float(kp[1])},
                                                   "score": float(keypoints_scores[j])}
            obj["keypoints"] = json_keypoints
            obj["length"] = None
            res.append(obj)
    return res


def test_stereo():
    stereo_dir = "data/Frames.paired"
    left_dir, right_dir = os.path.join(stereo_dir, "Left"), os.path.join(stereo_dir, "Right")
    model, transforms, device = load_model_eval("work_dir/epoch_8.state_dict.pth", is_state_dict=True, use_cpu=False)
    fnames = os.listdir(left_dir)
    for fname in fnames:
        left_image = cv2.imread(os.path.join(left_dir, fname))
        right_image = cv2.imread(os.path.join(right_dir, fname))
        start = time.time()
        left_res = predict(left_image, model, transforms, device, show=False)
        right_res = predict(right_image, model, transforms, device, show=False)
        left_res, right_res = cal_lengths_by_res(left_image, right_image, left_res, right_res)
        end = time.time()
        print("Time Consumed: {}".format(end - start))
        pass


if __name__ == '__main__':
    # train_and_eval(epochs=24, batch_size=1)
    # test_model("work_dir/epoch_8.state_dict.pth", "data/Frames", is_state_dict=True,
    #            box_score_thre=0.5, kp_score_thre=0,
    #            save_image_out=True, save_json_out=True)
    test_stereo()
