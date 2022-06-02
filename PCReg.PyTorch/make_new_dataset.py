import numpy as np 
import os 

# frag_1 = 'Cone_shard.028.npy'
# files = os.listdir("/home/sombit/Downloads/objects/5_shapes_4-1_seeds/Cone_16_seed_0")

dataset_path = '/home/sombit/Downloads/objects/dataset/'
dataset = os.listdir(dataset_path)
for data in dataset:
	arr = np.zeros((1,6))

	data_path = os.path.join(dataset_path,data)
	print(data)
	files = os.listdir(data_path)
	for file in files:

		file_path = os.path.join(data_path,file)
		if True:
			if(file.endswith('.npy')):
				x = np.load(file_path ,allow_pickle =True)
				arr = np.vstack((arr,x))
	arr = np.delete(arr,0,0)
	# print(data[0])
	print(arr.shape)
	np.save(os.path.join(data_path,"all.npy"),arr)