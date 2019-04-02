import sys, os, time
import torch

def create_vocfile(dset, vocabfile):
    if not os.path.exists(vocabfile):
        before = time.time()
        print('making caption file at {}'.format(vocabfile), flush=True)
        with open(vocabfile, 'w+') as f:
            for i, sample in enumerate(dset):
                caption = sample['caption']
                f.write(caption + '\n')
                if i % 10000 == 9999:
                    print('done adding {} sentences, {}s per entry'.format(i+1, (time.time()-before)/(i+1)), flush=True)
        print('created caption file, took {}s'.format(time.time()-before), flush=True)
    else:
        print('found existing caption file at {}'.format(vocabfile), flush=True)

def collate_fn(samples):
    images = []
    captions = []
    bboxes = []
    for sample in samples:
        im = sample['image']
        cap = sample['caption']
        bbox = sample['bboxinfo']
        images.append(im)
        captions.append(cap)
        bboxes.append(bbox)
    im_batch = torch.stack(images)
    return {'image' : im_batch, 'caption' : captions, 'bboxinfo' : bboxes}


def xyxy2xywh(x1, y1, x2, y2):
    x = (x1 + x2) // 2
    y = (y1 + y2) // 2
    w = x1 - x2
    h = y1 - y2
    return x, y, w, h

def xywh2xyxy(x, y, w, h):
    x1 = int(x + w/2)
    x2 = int(x - w/2)
    y1 = int(y + h/2)
    y2 = int(y - h/2)
    return x1, x2, y1, y2

# function to compute IOU over two batches of bounding boxes.
# input: (N, 4), (N, 4)
def compute_iou(bbox1, bbox2):
    b1_x, b1_y, b1_w, b1_h = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x, b2_y, b2_w, b2_h = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]
    b1_x1, b1_y1, b1_x2, b1_y2 = xywh2xyxy(b1_x, b1_y, b1_w, b1_h)
    b2_x1, b2_y1, b2_x2, b2_y2 = xywh2xyxy(b2_x, b2_y, b2_w, b2_h)

    inter_x1 = torch.min(b1_x1, b2_x1)
    inter_y1 = torch.min(b1_y1, b2_y1)
    inter_x2 = torch.max(b1_x2, b2_x2)
    inter_y2 = torch.max(b1_y2, b2_y2)

    intersection = (inter_x1 - inter_x2 + 1) * (inter_y1 - inter_y2 + 1)
    print(intersection)
    b1_area = (b1_x1 - b1_x2 + 1) * (b1_y1 - b1_y2 + 1)
    b2_area = (b2_x1 - b2_x2 + 1) * (b2_y1 - b2_y2 + 1)
    union = b1_area + b2_area - intersection
    return intersection / union

# finds the grid that is responsible for bounding box detection
# S: number of grids
# bbox: (4), x1y1x2y2
def get_grid_coord(S, bbox, imsize):
    # bin of pixels for each grid cell
    bin = imsize // S
    x_center = (bbox[2] + bbox[0]) / 2
    y_center = (bbox[3] + bbox[1]) / 2
    tar_x = int(x_center // bin)
    tar_y = int(y_center // bin)
    return tar_x, tar_y


# function to scale anchor widths and heights
# anchor : (B, 1)
def scale_anchors(imsize, S, anchor):
    bin = int(imsize / S)
    scaled = anchor / bin
    return scaled


# function to offset predicted bounding boxes by grid
# x, y, w, h, conf : (bs, B, S, S, 1)
# cls : (bs, B, S, S, *)
# scaled_anchor : (B, 2)
# returns : (bs, B, S, S, 1), xywh
def offset_boxes(x, y, w, h, device, imsize, S, scaled_anchor):
    anchor_num = x.size(1)
    bin = int(imsize / S)
    # offsets : (1, 1, S, S, 1)
    offset_x = torch.arange(S, dtype=x.dtype, device=device).repeat(S, 1).view(1, 1, S, S, 1)
    offset_y = offset_x.clone().transpose(0, 1)
    anchor_w, anchor_h = scaled_anchor[:, 0], scaled_anchor[:, 1]
    # scaled : (1, B, 1, 1)
    scaled_w = anchor_w.view(1, anchor_num, 1, 1, 1)
    scaled_h = anchor_h.view(1, anchor_num, 1, 1, 1)
    # pred : (bs, B, S, S, 1)
    pred_x = x.new_empty(x.size())
    pred_y = x.clone()
    pred_w = x.clone()
    pred_h = x.clone()
    print(w.size(), scaled_w.size())

    pred_x = x + offset_x
    pred_y = y + offset_y
    pred_w = torch.exp(w) * scaled_w
    pred_h = torch.exp(h) * scaled_h
    return pred_x, pred_y, pred_w, pred_h


# x, y, w, h, conf, cls : (bs, B, S, S, 1)
# anchors : (B, 2)
# confidence : (bs, 1)
# labels : (bs, C)
def create_targets(device, anchors, x, y, w, h, conf, cls, imsize=256):
    bs, B, S, _, _ = x.size()
    for batch in range(bs):
        for bbid in range(B):
            # bbox : (S*S, 4)
            # iou : (S*S)
            # maxim : (1)
            pass


if __name__ == '__main__':
    b1 = torch.tensor([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]])
    b2 = torch.tensor([[2.0, 2.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0]])
    print(compute_iou(b1, b2))
    output = torch.randn(2, 95, 7, 7)
    create_targets(output, bboxes, confidences, classes)

