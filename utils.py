import numpy as np
import random
import cv2
from tqdm import tqdm
import math

class datasource(object):
	def __init__(self, images, poses, names,images_cropped):
		self.images = images
		self.poses = poses
		self.names = names
		self.images_cropped = images_cropped
class vecsource(object):
	def __init__(self, vecs, poses):
		self.vecs = vecs
		self.poses = poses

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = int((new_height - output_side_length) / 2)
	width_offset = int((new_width - output_side_length) / 2)
#	pdb.set_trace()
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	cropped_images = []

	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		X = cv2.resize(X, (455, 256))
		X = centeredCrop(X, 224)
		images_cropped.append(X)
		X = X[...,::-1]
		cropped_images.append(X)

	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1)) 
		mean[0][0] += X[0,:,:]
		mean[0][1] += X[1,:,:]
		mean[0][2] += X[2,:,:]
		N += 1
	mean[0] /= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		Y = np.expand_dims(X, axis=0)
		images_out.append(Y)
			
	return images_out, cropped_images

def get_data(directory,dataset):
	poses = []
	images = []
	names = []
	with open(directory+dataset) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses.append((p0,p1,p2,p3,p4,p5,p6))
			images.append(directory+fname)
			names.append(fname)
	images,images_cropped = preprocess(images)
	return datasource(images, poses, names, images_cropped)

def gen_data(source):
	while True:
		indices = range(len(source.images))
		random.shuffle(indices)
		for i in indices:
			image = source.images[i]
			pose_x = source.poses[i][0:3]
			pose_q = source.poses[i][3:7]
			yield image, pose_x, pose_q

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)

        
def quaternion_to_euler(x, y, z, w):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z
