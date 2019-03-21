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

