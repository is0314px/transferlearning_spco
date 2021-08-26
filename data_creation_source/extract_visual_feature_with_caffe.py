#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, os.path, numpy, caffe, glob, csv
import caffe.io
import numpy as np

#FULL PATH
MEAN_FILE = '/home/ema/Caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = '/home/ema/Caffe/examples/imagenet/imagenet_feature.prototxt'
PRETRAINED = '/home/ema/Caffe/examples/imagenet/caffe_reference_imagenet_model'
#LAYER = 'fc6'
#LAYER = 'prob'
#LAYER = 'fc6wi'
#LAYER = 'fc7wi'
LAYER = 'fc7'#LAYERを変更することでどの層を抽出するか選べる
#層の記述はcaffe/examples/imagenet/imagenet_feature.prototxt
#参考 http://ataniguchi.hatenablog.com/entry/2016/07/14/202450
INDEX = 4

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
caffe.set_mode_cpu()
net.transformer.set_mean('data', numpy.load(MEAN_FILE))
net.transformer.set_raw_scale('data', 255)
net.transformer.set_channel_swap('data', (2,1,0))

dir = '../gibbs_dataset/sigverse/3LDK_9/'
S = 5
data_num = 240
start_num = 300

for i in range(data_num):
	#try:
	j = i + start_num
	img = dir+'image/'+repr(j)+'.png'
	Img = [img, ]

	for f in Img:
			image = caffe.io.load_image(f)
			net.predict([image])
			feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
			feat_numpy = np.array(feat)
			feat_numpy = feat_numpy * S #fc7のとき　Sは画像特徴の値の範囲を0~100にする正規化項
			#feat_numpy = feat_numpy * 100 #最終層のとき、100倍
			#feat_numpy = np.round(feat_numpy) #小数点以下四捨五入
			feat_numpy = feat_numpy.astype(int) #int変換
			feat = feat_numpy.tolist()

	with open(dir+'vision_fc7_normalized/'+repr(j)+'.csv', 'a') as f_handle:
		#writer = csv.writer(f_handle,delimiter="\t",lineterminator='\n') # 改行コード（\n）を指定しておく
		writer = csv.writer(f_handle,delimiter="\n",lineterminator='\n')
		writer.writerow(feat)
			
	#except:
	#	pass

