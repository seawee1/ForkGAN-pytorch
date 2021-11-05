from pycocotools.coco import COCO
import torch.utils.data as data
from PIL import Image
import os
import copy
from PIL import ImageDraw

class UnalignedCocoDataset(data.Dataset):
    """
    This dataset is the analogous version of unaligned_dataset but adopted to COCO dataset format.
    It is assumed that two COCO .jsons (either trainA/trainB or testA/testB) are located under opt.dataroot (depending on opt.phase).
    You can further specify the image root directory of the COCO dataset via opt.coco_imagedir.
    Cropping transformation attempts to preserve at least the first objects in the COCO annots list.
    This dataset returns the first (and only the first) object bbox of the specified category_id.
    """
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.coco_imagedir = opt.coco_imagedir
        # The category_id of objects to consider. At the moment only single object category is supported.
        self.category_id = opt.category_id

        self.annotA = os.path.join(opt.dataroot, opt.phase + 'A.json')
        self.annotB = os.path.join(opt.dataroot, opt.phase + 'B.json')
        self.cocoA = COCO(self.annotA)
        self.cocoB = COCO(self.annotB)
        self.idsA = list(sorted(self.cocoA.imgs.keys()))
        self.idsB = list(sorted(self.cocoB.imgs.keys()))

        self.A_size = len(self.idsA)
        self.B_size = len(self.idsB)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        indexA = index % self.A_size
        if self.opt.serial_batches:   # make sure index is within then range
            indexB = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            indexB = random.randint(0, self.B_size - 1)

        # COCO object
        cocoA = self.cocoA
        cocoB = self.cocoB
        # Image IDs
        img_idA = self.idsA[indexA]
        img_idB = self.idsB[indexB]
        # pycocotools has problems with image ids of type str. Thus, pack them into lists if necessary.
        if not isinstance(img_idA, int) or not isinstance(img_idA, list):
            img_idA = [img_idA]
            img_idB = [img_idB]
        # Annot IDs
        ann_idsA = cocoA.getAnnIds(imgIds=img_idA)
        ann_idsB = cocoB.getAnnIds(imgIds=img_idB)
        # Annotation lists
        annotsA = copy.deepcopy(cocoA.loadAnns(ann_idsA))
        annotsB = copy.deepcopy(cocoB.loadAnns(ann_idsB))
        # Image paths
        pathA = cocoA.loadImgs(img_idA)[0]['file_name']
        pathB = cocoB.loadImgs(img_idB)[0]['file_name']
        # Open images
        A = Image.open(os.path.join(self.root, self.coco_imagedir, pathA)).convert('RGB')
        B = Image.open(os.path.join(self.root, self.coco_imagedir, pathB)).convert('RGB')
        # Filter out wrong objects
        annotsA = filter_cat_ids(self.category_id, annotsA)
        annotsB = filter_cat_ids(self.category_id, annotsB)
        # Convert xywh to x1y1x2y2 bboxes
        annotsA = bbox_wh_to_xy(annotsA)
        annotsB = bbox_wh_to_xy(annotsB)
        # Transform images and annots
        A, annotsA = self.transform_A(A, annotsA)
        B, annotsB = self.transform_B(B, annotsB)
        # None type entries have to be removed from annots dicts
        #annotsA = annots_rm_none(annotsA)
        #annotsB = annots_rm_none(annotsB)

        # Only return first annotation, place dummy bbox [-1, -1, -1, -1] if no bbox found.
        if len(annotsA) != 0:
            bboxA = annotsA[0]['bbox']
        else:
            bboxA = [-1, -1, -1, -1]
        if len(annotsB) != 0:
            bboxB = annotsB[0]['bbox']
        else:
            bboxB = [-1, -1, -1, -1]

        return {'A': A, 'B': B, 'A_paths': pathA, 'B_paths': pathB, 'A_bboxes': torch.Tensor(bboxA), 'B_bboxes': torch.Tensor(bboxB)}

    def __len__(self):
        return max(self.A_size, self.B_size)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--coco_imagedir', required=False, help='path to COCO image folder')
        parser.add_argument('--category_id', default=1, type=int, help='Category id of target object class')
        return parser

# Converts xywh to x1y1x2y2 bboxes
def bbox_wh_to_xy(annots):
    for annot in annots:
        annot['bbox'][2] = annot['bbox'][0] + annot['bbox'][2]
        annot['bbox'][3] = annot['bbox'][1] + annot['bbox'][3]
    return annots

# Converts None dict values to -1 values
def annots_rm_none(annots):
    for annot in annots:
        for k, v in annot.items():
            if v is None:
                annot[k] = -1
    return annots

# Filter annotations based on cat id
def filter_cat_ids(cat_id, annots):
    ret_annots = []
    for annot in annots:
        if annot['category_id'] == cat_id:
            ret_annots.append(annot)
    return ret_annots

# Get transform based on opt.preprocess
def get_transform(opt, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(ResizeSide(opt.load_size, 'width', method))
    elif 'scale_height' in opt.preprocess:
        transform_list.append(ResizeSide(opt.load_size, 'height', method))
    if 'crop' in opt.preprocess:
        transform_list.append(RandomCrop(opt.crop_size))

    if opt.preprocess == 'none':
        transform_list.append(Power2(base=4, method=method))

    if not opt.no_flip:
        transform_list.append(RandomHorizontalFlip())
    if convert:
        transform_list += [ToTensor()]
        if grayscale:
            transform_list += [Normalize((0.5,), (0.5,))]
        else:
            transform_list += [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return Compose(transform_list)

##############################################################################
# Annotation transform helpers
##############################################################################

# Resize bbox annotations based on old and new image dimensions
def resize_annots(annots, ow, oh, w, h):
    sw = w / ow # Scale factor width
    sh = h / oh # Scale factor height
    for annot in annots:
        annot['bbox'][0] = int(round(annot['bbox'][0] * sw))
        annot['bbox'][1] = int(round(annot['bbox'][1] * sh))
        annot['bbox'][2] = int(round(annot['bbox'][2] * sw))
        annot['bbox'][3] = int(round(annot['bbox'][3] * sh))
    return annots

# Adjust bbox coordinates based on crop area given by (x, y, th, tw)
def crop_annots(annots, x, y, tw, th):
    annots_cropped = []
    for annot in annots:
        # Test if annot outside crop area
        if annot['bbox'][0] >= x + tw or annot['bbox'][1] >= y + th or annot['bbox'][2] <= x or annot['bbox'][3] <= y:
            continue
        # Calculate bbox coordinates. Allow partial bbox crops
        annot['bbox'][0] = max(0, annot['bbox'][0] - x)
        annot['bbox'][1] = max(0, annot['bbox'][1] - y)
        annot['bbox'][2] = min(tw, annot['bbox'][2] - x)
        annot['bbox'][3] = min(th, annot['bbox'][3] - y)
        annots_cropped.append(annot)
    return annots_cropped

# Flip annotation x or y coordinates. Either 'horizontal' or 'vertical'
def flip_annots(img, annots, ax='horizontal'):
    w, h = img.size
    for annot in annots:
        if ax == 'horizontal':
            x2 = w - annot['bbox'][0]
            x1 = w - annot['bbox'][2]
            annot['bbox'][0] = x1
            annot['bbox'][2] = x2
        else:
            y2 = h - annot['bbox'][1]
            y1 = h - annot['bbox'][3]
            annot['bbox'][1] = y1
            annot['bbox'][3] = y2
    return annots

##############################################################################
# Custom Transforms working with annots
##############################################################################

import numbers
import random
from collections.abc import Sequence
import torch
from torchvision.transforms import functional as F
from enum import Enum

class InterpolationMode(Enum):
    """Interpolation modes
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, annots):
        for t in self.transforms:
            img, labels = t(img, annots)
        return img, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Power2:
    def __init__(self, base, method=Image.BICUBIC):
        self.base = base
        self.method = method

    def __call__(self, img, annots):
        ow, oh = img.size
        h = int(round(oh / self.base) * self.base)
        w = int(round(ow / self.base) * self.base)
        if h == oh and w == ow:
            return img, annots
        return img.resize((w, h), self.method), resize_annots(annots, ow, oh, w, h)


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor:
    def __call__(self, pic, annots):
        return F.to_tensor(pic), annots

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor, annots):
        return F.normalize(tensor, self.mean, self.std, self.inplace), annots

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Grayscale(torch.nn.Module):
    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, img, annots):
        return F.rgb_to_grayscale(img, num_output_channels=self.num_output_channels), annots

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)

class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size

        self.interpolation = interpolation

    def forward(self, img, annots):
        if isinstance(self.size, int):
            h, w = self.size, self.size
        else:
            h, w = self.size
        ow, oh = img.size
        return F.resize(img, self.size, self.interpolation), resize_annots(annots, ow, oh, w, h)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class RandomCrop(torch.nn.Module):
    # Get crop window. Always try to fully preserve first object in annots list
    @staticmethod
    def get_params(img, output_size, bbox):
        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        xmin = max(0, bbox[2] - tw)
        xmax = min(w - tw, bbox[0])
        ymin = max(0, bbox[3] - th)
        ymax = min(h - th, bbox[1])

        # Hacky
        if xmin > xmax:
            xmin = 0
            xmax = w - tw
        if ymin > ymax:
            ymin = 0
            ymax = h - th

        i = torch.randint(ymin, ymax + 1, size=(1, )).item()
        j = torch.randint(xmin, xmax + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", ):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, annots):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            for annot in annots:
                if len(self.padding) >= 1:
                    annot['bbox'][0] += self.padding[0]
                    annot['bbox'][2] += self.padding[0]
                if len(self.padding) >= 3:
                    annot['bbox'][1] += self.padding[2]
                    annot['bbox'][3] += self.padding[2]

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            annot['bbox'][0] += self.size[1] - width
            annot['bbox'][2] += self.size[1] - width
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            annot['bbox'][1] += self.size[0] - height
            annot['bbox'][3] += self.size[0] - height

        # Take first bbox
        if len(annots) == 0:
            bbox = [img.width, img.height, 0, 0] # Hacky
        else:
            bbox = annots[0]['bbox']
        i, j, h, w = self.get_params(img, self.size, bbox)

        return F.crop(img, i, j, h, w), crop_annots(annots, j, i, w, h)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, annots):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            annots = flip_annots(img, annots, ax='horizontal')
        return img, annots

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Lambda:
    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def __call__(self, img, annots):
        return self.lambd(img, annots)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ResizeSide:
    def __init__(self, size, side='height', interpolation=Image.BICUBIC):
        self.size = size
        self.side = side
        self.interpolation = interpolation

    def __call__(self, img, annots):
        ow, oh = img.size
        if self.side == 'width':
            w = self.size
            h = int((self.size / ow) * oh)
        else:
            w = int((self.size / oh) * ow)
            h = self.size
        return img.resize((w, h), self.interpolation), resize_annots(annots, ow, oh, w, h)

# For debugging
class DrawBBox:
    def __call__(self, img, annots):
        draw = ImageDraw.Draw(img)
        for annot in annots:
            draw.rectangle([(annot['bbox'][0], annot['bbox'][1]),(annot['bbox'][2],annot['bbox'][3])], outline='red')
        return img, annots