import mxnet as mx
import numpy as np
import os
import json
import cv2
#from ava.utils import utils
from collections import namedtuple

ctx = [mx.gpu(0)]
Batch = namedtuple('Batch',['data'])

def load_fine_tune_symbol(prefix,layer_name,output_layer_num):
        sym=mx.symbol.load('%s-symbol.json'%prefix)
        all_layers=sym.get_internals()
        net = all_layers[layer_name+'_output']
        net = mx.symbol.FullyConnected(data=net, num_hidden=output_layer_num, name='fc1_new')
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
        sym = net
        sym.save('%s-fixed-symbol.json'%prefix)
        return sym

def load_params(prefix,epoch):
        save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
        arg_params = {}
        aux_params = {}
        for k,v in save_dict.items():
                tp,names=k.split(':',1)
                if tp == 'arg':
                        arg_params[names] = v
                if tp == 'aux':
                        aux_params[names] = v
        return arg_params, aux_params


def load_image(path):
        mean_r, mean_g, mean_b=float(123.68),float(116.779),float(103.939)
        std_r, std_g, std_b = float(58.395),float(57.12),float(57.375)
        mean_img = mx.nd.load('/workspace/data/trainmean.bin').values()[0].asnumpy()
        origimg= cv2.imread(path)
        img = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
        img = img.astype(float)
        img = cv2.resize(img, (224,224))
        #img -= [mean_r, mean_g, mean_b]
        #img /= [std_r, std_g, std_b]
        img = np.swapaxes(img,0,2)
        img = np.swapaxes(img,1,2)
        img=img-mean_img
        img = img[np.newaxis,:]
        return img

def main():
        filename = os.listdir('/workspace/data/test')
        with open('/workspace/examples/mxnet/match_name.json','r') as f:
            name_match=json.load(f)
        sym,arg_params,aux_params=mx.model.load_checkpoint(prefix='/workspace/run/5a4ed884ac2c3d00016baf69/output',epoch=40)

        labels=["american","chinese","european","japanese","mediterranean","modern","new_classic","north_european","rural","southeast_asian"]
        mod=mx.mod.Module(symbol=sym,context=ctx)
        mod.bind(for_training=False,data_shapes=[('data',(1,3,224,224))],
                label_shapes=mod._label_shapes)

        mod.set_params(arg_params,aux_params,allow_missing=True)

        out_list=[]
        for i in filename:
            try:
                img=load_image('/workspace/data/test/'+i)
            except (cv2.error):
                print ('Skipping image: '+ i)
                continue
            mod.forward(Batch([mx.nd.array(img)]))
            prob=mod.get_outputs()[0].asnumpy()
            prob = np.squeeze(prob)
            a= np.argsort(prob)[::-1]
            url=name_match[i]
            out_dict={}
            out_dict.update({"name":url})
            label=url.rsplit('/',2)[1]
            out_dict.update({"label":label})
            index=1
            for i in a[0:1]:
                    out_dict.update({str(index):{'class':labels[i],'probability':str(prob[i])}})
                    index=index+1
            out_list.append(out_dict)

        with open('output.json','w') as op:
            for i in out_list:
                op.write('{}\n'.format(json.dumps(i)))

if __name__ == '__main__':
        main()