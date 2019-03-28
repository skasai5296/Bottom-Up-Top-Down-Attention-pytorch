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
    x = (x2 + x1) // 2
    y = (y2 + y1) // 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def xywh2xyxy(x, y, w, h):
    x1 = int(x - w/2)
    x2 = int(x + w/2)
    y1 = int(y - h/2)
    y2 = int(y + h/2)
    return x1, x2, y1, y2

# function to compute IOU over two batches of bounding boxes.
# input: (N, 4), (N, 4)
def compute_iou(bbox1, bbox2):
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    inter_x1 = torch.min(b1_x1, b2_x1)
    inter_y1 = torch.min(b1_y1, b2_y1)
    inter_x2 = torch.max(b1_x2, b2_x2)
    inter_y2 = torch.max(b1_y2, b2_y2)

    intersection = (inter_x1 - inter_x2 + 1) * (inter_y1 - inter_y2 + 1)
    print(intersection)
    b1_area = (b1_x1 - b1_x2 + 1) * (b1_y1 - b1_y2 + 1)
    print(b1_area)
    b2_area = (b2_x1 - b2_x2 + 1) * (b2_y1 - b2_y2 + 1)
    print(b2_area)
    union = b1_area + b2_area - intersection
    print(union)
    return intersection / union

# finds the grid that is responsible for bounding box detection
# S: number of grids
# bbox: (4)
def get_grid_coord(S, bbox, imsize):
    # bin of pixels for each grid cell
    bin = imsize // S
    x_center = (bbox[2] + bbox[0]) / 2
    y_center = (bbox[3] + bbox[1]) / 2
    tar_x = int(x_center // bin)
    tar_y = int(y_center // bin)
    return tar_x, tar_y

# pred : (bs, B, 4, S, S)
# anchor : (B, 4)
# confidence : (bs, 1)
# labels : (bs, C)
def create_target(S, anchor, pred, confidence, labels):
    bs, B, _, S, _ = pred.size()
    # pred : (bs, B, S*S, 4)
    pred = pred.flatten(start_dim=-1).transpose(2, 3)
    for batch in range(bs):
        for bbid in range(B):
            # bbox : (S*S, 4)
            bbox = pred[bs, bbid, :, :]
            anc = anchor[bbid, :].expand_as(bbox)
            # iou : (S*S)
            iou = compute_iou(bbox, anc)
            # maxim : (1)
            maxim = torch.argmax(iou)
            tar_x = maxim % S
            tar_y = maxim // S
            sx, sy = get_grid_coord(bbox[idx])
            mask[:, bbid, sx, sy] = 1
            conf[:, bbid, sx, sy]


if __name__ == '__main__':
    b1 = torch.tensor([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]])
    b2 = torch.tensor([[2.0, 2.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0]])
    print(compute_iou(b1, b2))

    output = torch.randn(2, 95, 7, 7)
    make_targets(output, bboxes, confidences, classes)

