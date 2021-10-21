# -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template
#from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from predict import infer_classify, infer_classify_whole_image, infer_classify_whole_image_v2
from flask import jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import torch
from torchvision import models
import torch.nn as nn 
import pdb
import uuid
import glob
from utils import remove_files, get_list_files 


def load_model_classify():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 17)
    #model = model.to('cuda')
    model.load_state_dict(torch.load("./checkpoint_new_dataset_bb50_lan2.pth", map_location='cpu'))
    model.eval()
    class_name = ['ASC_H', 'ASC_US', 'Atophy', 'Bach_cau_da_nhan', 'Dam_te_bao_HSIL', 'Dam_te_bao_LSIL', 'Di_san_vay', 'TE_BAO_LSIL', 'Te_bao_HSIL', 'Te_bao_can_day', 'Te_bao_day', 'Te_bao_noi_mac', 'Te_bao_tuyen_co_trong', 'Te_bao_vay_be_mat', 'Te_bao_vay_trung_gian', 'Ung_thu_bieu_mo_te_bao_vay', 'background']
    return model, class_name


app = Flask(__name__)
CORS(app)

Te_bao_BatThuong = ['ASC_H', 'ASC_US', 'Dam_te_bao_HSIL', 'Dam_te_bao_LSIL', 'TE_BAO_LSIL', 'Te_bao_HSIL', 'Ung_thu_bieu_mo_te_bao_vay']

Te_bao_BinhThuong = ['Atophy', 'Bach_cau_da_nhan', 'Di_san_vay', 'Te_bao_can_day', 'Te_bao_day', 'Te_bao_noi_mac', 'Te_bao_tuyen_co_trong', 'Te_bao_vay_be_mat', 'Te_bao_vay_trung_gian']


#model, config = load_model() # mo hinh nhan biet nhan te bao
model_classify, class_name = load_model_classify()


#@app.route('/api', methods=['GET', 'POST'])
#def api():
#    img_request = request.files['image']
#    img_file = secure_filename(img_request.filename)
#    img_request.save(os.path.join("./static/images/", img_file))
#    print(img_file)
#    with graph.as_default():
#        predict_path = infer(model, config, img_file)
#    result = {
#            "bb_path": predict_path,    
#        }
#    return jsonify(result)

@app.route('/api_classify', methods=['GET', 'POST'])
def api_classify():
    img_request = request.files['image']
    img_file = secure_filename(img_request.filename)
    img_request.save(os.path.join("./static/images/", img_file))
    print(img_file)
    classify_name = infer_classify(img_file, model_classify, class_name)
    result = {
            "class_name": classify_name,    
        }
    return jsonify(result)

@app.route('/classify_whole_image', methods=['GET', 'POST'])
def classify_whole_image():
    img_request = request.files['image']
    img_file = secure_filename(img_request.filename)
    img_request.save(os.path.join("./static/images/", img_file))
    print(img_file)
    #result = infer_classify_whole_image(img_file, model_classify, class_name, Te_bao_BatThuong, Te_bao_BinhThuong)
    result = infer_classify_whole_image_v2(img_file, model_classify, class_name, Te_bao_BatThuong, Te_bao_BinhThuong)
    return jsonify(result)

@app.route('/data_labeling', methods=['GET', 'POST'])
def data_labeling():
    try:
        img_request = request.files['image']
        img_label = request.form.get('label')
        uuid_name = str(uuid.uuid1()).replace("-","")+".png"

        path_save = os.path.join("./static/data_labeling", img_label)
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        img_request.save(os.path.join(path_save,uuid_name))
        return {
                "name": uuid_name,
                "status": "Done"
            }
    except Exception as e:
        return {
                "status": "Fail"
            }
@app.route('/delete_label', methods=['GET', 'POST'])
def delete_label():
    try:
        label_delete = request.form.get('name')
        all_file = glob.glob('./static/data_labeling/*/*') 
        has_file_delete = remove_files(all_file, label_delete)
        return {
                "Delete": has_file_delete,
            }
    except Exception as e:
        print(e)
        return {
                "status": "Fail"
            }

@app.route('/get_list_label', methods=['GET', 'POST'])
def get_list_label():
    try:
        label = request.form.get('label')
        list_labels = get_list_files(label)
        return {
                "data": list_labels
            }
    except Exception as e:
        print(e)
        return {
                "status": "Fail"
            }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='50055')
