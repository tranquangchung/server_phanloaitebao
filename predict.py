"""
Prediction part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import os
from PIL import Image
import pdb

transform = transforms.Compose([
    transforms.Resize((352,352)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

data_transforms_classify = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


threshold = 0.5

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def infer_classify(img_path, model, class_name):
    img_path = os.path.join("./static/images", img_path)
    image = rgb_loader(img_path)
    width, height = image.size
    image = data_transforms_classify(image).unsqueeze(0)
    image = image.cuda()
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return class_name[preds[0]]

def infer_classify_whole_image(img_path, model, class_name, Te_bao_BatThuong, Te_bao_BinhThuong):
    def convert_to_tensor(cells):
        cell_transforms = []
        for cell in cells:
            cell = Image.fromarray(np.uint8(cell)).convert('RGB')
            cell_transform = data_transforms_classify(cell)
            cell_transforms.append(cell_transform)
        return torch.stack(cell_transforms)

    img_path = os.path.join("./static/images", img_path)
    img = cv2.imread(img_path) 
    h, w, _ = img.shape
    step = 128
    size = 128
    margin = 0 #step
    total_cells = []
    for y in range(0, h-margin, step):
        for x in range(0, w-margin, step):
            cell = img[y:y+size, x:x+size]
            h_tmp, w_tmp, _ = cell.shape
            #if h_tmp < size or w_tmp < size:
            #    continue
            total_cells.append(cell)
    data = convert_to_tensor(total_cells) 
    print("*"*20)
    print(data.shape)
    with torch.no_grad(): 
        data = data.to('cuda')
        outputs = model(data)
        outputs_tanh = torch.tanh(outputs)
        _, preds = torch.max(outputs, 1)
    tebao_bat_thuong = []
    tebao_binh_thuong = []
    index = 0
    for y in range(0, h-margin, step):
        for x in range(0, w-margin, step):
            cell = img[y:y+size, x:x+size]
            h_tmp, w_tmp, _ = cell.shape
            #if h_tmp < size or w_tmp < size:
            #    continue
            tebao = class_name[preds[index]]
            score = outputs_tanh[index,preds[index]]
            if score.item() > 0.95:
                #print(tebao)
                if tebao == 'background':
                    pass
                elif tebao in Te_bao_BatThuong:
                    object_tebao = {
                            "name": tebao,
                            "x": x,
                            "y": y,
                            "width": w_tmp,
                            "height": h_tmp
                            }
                    tebao_bat_thuong.append(object_tebao)
                elif tebao in Te_bao_BinhThuong:
                    object_tebao = {
                            "name": tebao,
                            "x": x,
                            "y": y,
                            "width": w_tmp,
                            "height": h_tmp
                            }
                    tebao_binh_thuong.append(object_tebao)
            else:
                print("nhay vao day")
            index += 1
    results = {
            "binh_thuong": tebao_binh_thuong,
            "bat_thuong": tebao_bat_thuong,
            }
    return results

def infer_classify_whole_image_v2(img_path, model, class_name, Te_bao_BatThuong, Te_bao_BinhThuong):
    def convert_to_tensor(cells):
        cell_transforms = []
        for cell in cells:
            cell = Image.fromarray(np.uint8(cell)).convert('RGB')
            cell_transform = data_transforms_classify(cell)
            cell_transforms.append(cell_transform)
        return torch.stack(cell_transforms)

    img_path = os.path.join("./static/images", img_path)
    img = cv2.imread(img_path) 
    h, w, _ = img.shape
    step = 128
    size = 128
    margin = 0 #step
    total_cells = []
    for y in range(0, h-margin, step):
        for x in range(0, w-margin, step):
            cell = img[y:y+size, x:x+size]
            h_tmp, w_tmp, _ = cell.shape
            if h_tmp < (size/2) or w_tmp < (size/2):
                continue
            total_cells.append(cell)
    data = convert_to_tensor(total_cells) 
    with torch.no_grad(): 
        data = data.to('cuda')
        outputs = model(data)
        outputs_tanh = torch.tanh(outputs)
        _, preds = torch.max(outputs, 1)
    tebao_bat_thuong = []
    tebao_binh_thuong = []
    index = 0
    mask_image = np.zeros((h,w, len(class_name)), dtype='uint8')
    for y in range(0, h-margin, step):
        for x in range(0, w-margin, step):
            cell = img[y:y+size, x:x+size]
            h_tmp, w_tmp, _ = cell.shape
            if h_tmp < (size/2) or w_tmp < (size/2):
                continue
            #cv2.imwrite("./mask_toa_do/anh_{0}.png".format(str(index)), cell)
            tebao = class_name[preds[index]]
            score = outputs_tanh[index,preds[index]]
            #mask_image[y:y+size, x:x+size, class_name.index(tebao)] = 255
            if score.item() > 0.95:
                mask_image[y:y+size, x:x+size, class_name.index(tebao)] = 255
            index += 1

    for i in range(len(class_name)):
        gray = mask_image[:,:,i]
        if np.max(gray)==0:
            continue
        name = "./mask_toa_do/{0}.png".format(class_name[i])
        cv2.imwrite(name, gray)
        tebao = class_name[i]
        gray = cv2.erode(gray, (5,5))
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            object_tebao = {
                    "name": tebao,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                    }
            if tebao in Te_bao_BatThuong:
                tebao_bat_thuong.append(object_tebao)
            elif tebao in Te_bao_BinhThuong:
                tebao_binh_thuong.append(object_tebao)

    results = {
            "binh_thuong": tebao_binh_thuong,
            "bat_thuong": tebao_bat_thuong,
            }
    return results
if __name__ == '__main__':

    import time

    start = time.time()

    # Create model configuration in inference mode
    config = KaggleBowlConfig()
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = 1
    config.display()

    model = get_model(config, model_path='../weight/kaggle_bowl.h5')

    infer(model, config, test_path='./abc.jpg')
    print("count: " , count)
