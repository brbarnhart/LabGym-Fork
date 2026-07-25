'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext.

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''


# Standard library imports.
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import itertools
import os
import random

# Related third party imports.
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import exposure,transform
from skimage.transform import AffineTransform
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras  # pylint: disable=unused-import
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers import (
	Activation,
	Add,
	AveragePooling2D,
	BatchNormalization,
	Conv2D,
	Dense,
	Dropout,
	Flatten,
	Input,
	LSTM,
	MaxPooling2D,
	TimeDistributed,
	ZeroPadding2D,
	concatenate,
	)
from keras.models import (
	Model,
	load_model,
	)
from keras.optimizers import SGD
from keras.utils import (
	Sequence,
	img_to_array,
	)

# Local application/library specific imports.
from LabGym.augment_export import (
	augment_export_task,
	augment_one_example,
	default_aug_workers,
	init_augment_worker,
	resolve_aug_methods,
)


matplotlib.use('Agg')


class DatasetFromPath_AA(Sequence):

	'''
	Load batches of training examples (including animations) from path
	'''

	def __init__(self,path_to_examples,length=15,batch_size=32,dim_tconv=16,dim_conv=32,channel=1,label_mode='hard_only',class_means=None):

		self.path_to_examples=path_to_examples
		self.length=length
		self.batch_size=batch_size
		self.dim_tconv=dim_tconv
		self.dim_conv=dim_conv
		self.channel=channel
		self.label_mode=label_mode
		self.class_means=class_means
		self.pattern_image_paths,self.classmapping=self.load_info()
		self.classnames=list(self.classmapping.keys())


	def load_info(self):

		pattern_image_paths=[]
		classnames=[]

		for pattern_image in os.listdir(self.path_to_examples):
			if pattern_image.endswith('.jpg'):
				pattern_image_paths.append(os.path.join(self.path_to_examples,pattern_image))
				classname=pattern_image.split('.jpg')[0].split('_')[-1]
				if classname not in classnames:
					classnames.append(classname)

		np.random.shuffle(pattern_image_paths)

		classnames.sort()
		labels=np.array(classnames)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		labels=[list(i) for i in labels]
		classmapping={name:labels[i] for i,name in enumerate(classnames)}

		return pattern_image_paths,classmapping


	def __len__(self):

		return int(np.floor(len(self.pattern_image_paths)/self.batch_size))


	def __getitem__(self,idx):

		batch=self.pattern_image_paths[idx*self.batch_size:(idx+1)*self.batch_size]
		animations=[]
		pattern_images=[]
		labels=[]

		for path_to_pattern_image in batch:

			animation=deque([np.zeros((self.dim_tconv,self.dim_tconv,self.channel),dtype='uint8')],maxlen=self.length)*self.length
			capture=cv2.VideoCapture(path_to_pattern_image.split('.jpg')[0]+'.avi')
			while True:
				retval,frame=capture.read()
				if frame is None:
					break
				if self.channel==1:
					frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				frame=cv2.resize(frame,(self.dim_tconv,self.dim_tconv),interpolation=cv2.INTER_AREA)
				animation.append(img_to_array(frame))
			animations.append(np.array(animation))

			pattern_image=cv2.imread(path_to_pattern_image)
			pattern_image=cv2.resize(pattern_image,(self.dim_conv,self.dim_conv),interpolation=cv2.INTER_AREA)
			pattern_images.append(img_to_array(pattern_image))

			labels.append(np.array(self.classmapping[path_to_pattern_image.split('.jpg')[0].split('_')[-1]]))

		animations=np.array(animations)
		animations=animations.astype('float32')/255.0
		pattern_images=np.array(pattern_images)
		pattern_images=pattern_images.astype('float32')/255.0
		labels=np.array(labels)
		labels=apply_class_mean_soft_to_labels(
			labels,self.classnames,self.class_means,self.label_mode)

		return [animations,pattern_images],labels



class DatasetFromPath(Sequence):

	'''
	Load batches of training examples (not including animations) from path
	'''

	def __init__(self,path_to_examples,batch_size=32,dim_conv=32,channel=3,label_mode='hard_only',class_means=None):

		self.path_to_examples=path_to_examples
		self.batch_size=batch_size
		self.dim_conv=dim_conv
		self.channel=channel
		self.label_mode=label_mode
		self.class_means=class_means
		self.pattern_image_paths,self.classmapping=self.load_info()
		self.classnames=list(self.classmapping.keys())


	def load_info(self):

		pattern_image_paths=[]
		classnames=[]

		for pattern_image in os.listdir(self.path_to_examples):
			if pattern_image.endswith('.jpg'):
				pattern_image_paths.append(os.path.join(self.path_to_examples,pattern_image))
				classname=pattern_image.split('.jpg')[0].split('_')[-1]
				if classname not in classnames:
					classnames.append(classname)

		np.random.shuffle(pattern_image_paths)

		classnames.sort()
		labels=np.array(classnames)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		labels=[list(i) for i in labels]
		classmapping={name:labels[i] for i,name in enumerate(classnames)}

		return pattern_image_paths,classmapping


	def __len__(self):

		return int(np.floor(len(self.pattern_image_paths)/self.batch_size))


	def __getitem__(self,idx):

		batch=self.pattern_image_paths[idx*self.batch_size:(idx+1)*self.batch_size]
		pattern_images=[]
		labels=[]

		for path_to_pattern_image in batch:

			pattern_image=cv2.imread(path_to_pattern_image)
			if self.channel==1:
				pattern_image=cv2.cvtColor(pattern_image,cv2.COLOR_BGR2GRAY)
			pattern_image=cv2.resize(pattern_image,(self.dim_conv,self.dim_conv),interpolation=cv2.INTER_AREA)
			pattern_images.append(img_to_array(pattern_image))

			labels.append(np.array(self.classmapping[path_to_pattern_image.split('.jpg')[0].split('_')[-1]]))

		pattern_images=np.array(pattern_images)
		pattern_images=pattern_images.astype('float32')/255.0
		labels=np.array(labels)
		labels=apply_class_mean_soft_to_labels(
			labels,self.classnames,self.class_means,self.label_mode)

		return pattern_images,labels


def apply_class_mean_soft_to_labels(hard_Y,classnames,class_means,label_mode):
	'''Build stacked [hard|soft] targets; soft from class means or hard copy.'''
	from LabGym.training.losses import maybe_stack_soft_targets

	hard=np.asarray(hard_Y,dtype=np.float32)
	if hard.ndim==1:
		if hard.shape[0]>0 and hard.max()<=1.0 and len(classnames)==2:
			hard=hard.reshape(-1,1)
		else:
			C=len(classnames)
			oh=np.zeros((hard.shape[0],C),dtype=np.float32)
			for i,v in enumerate(hard.astype(int)):
				if 0<=v<C:
					oh[i,v]=1.0
			hard=oh
	soft=hard.copy()
	if class_means and str(label_mode)!='hard_only':
		C=hard.shape[1]
		for i in range(hard.shape[0]):
			if C==1 and len(classnames)==2:
				idx=1 if hard[i,0]>=0.5 else 0
				cname=classnames[idx] if idx<len(classnames) else classnames[0]
			else:
				idx=int(np.argmax(hard[i]))
				cname=classnames[idx]
			if cname in class_means:
				m=class_means[cname]
				if C==1 and len(m)==2:
					soft[i,0]=m[1]
				elif len(m)==C:
					soft[i]=m
	return maybe_stack_soft_targets(hard,soft,label_mode)


class Categorizers():

	def __init__(self):

		self.extension_image=('.png','.PNG','.jpeg','.JPEG','.jpg','.JPG','.tiff','.TIFF','.bmp','.BMP') # the image formats that LabGym can accept
		self.extension_video=('.avi','.mpg','.wmv','.mp4','.mkv','.m4v','.mov') # the video formats that LabGym can accept
		self.classnames=None # the behavior category names in the trained Categorizer
		self.log=[]
		self.label_mode='hard_only'  # hard_only | hard_soft_aux | soft_primary
		self.lambda_soft=0.4


	def _resolve_soft_labels_path(self,data_path,soft_labels_path=None):
		if soft_labels_path and os.path.isfile(soft_labels_path):
			return soft_labels_path
		candidate=os.path.join(data_path,'soft_labels.csv')
		if os.path.isfile(candidate):
			return candidate
		return None


	def _soft_matrix_for_paths(self,path_files,classnames,data_path,soft_labels_path=None):
		'''Load soft label vectors aligned to path_files; None if unavailable.'''
		path=self._resolve_soft_labels_path(data_path,soft_labels_path)
		if path is None:
			return None
		try:
			from LabGym.training.soft_labels import SoftLabelTable
			table=SoftLabelTable.load_csv(path)
			basenames=[os.path.splitext(os.path.basename(p))[0] for p in path_files]
			return table.soft_matrix(basenames,classnames=list(classnames))
		except Exception as exc:
			print('Could not load soft labels from '+str(path)+': '+str(exc))
			self.log.append('Could not load soft labels: '+str(exc))
			return None


	def _compile_model(self,model,label_mode='hard_only',lambda_soft=0.4):
		from LabGym.training.losses import compile_with_label_mode
		return compile_with_label_mode(
			model,
			self.classnames,
			label_mode=label_mode,
			lambda_soft=lambda_soft,
			)


	def _stack_soft_targets(self,hard_labels,soft_matrix,label_mode):
		from LabGym.training.losses import maybe_stack_soft_targets
		return maybe_stack_soft_targets(hard_labels,soft_matrix,label_mode)


	def _class_mean_soft(self,data_path,classnames,soft_labels_path=None):
		'''Per-class mean soft vector from soft_labels.csv (fallback: None).'''
		path=self._resolve_soft_labels_path(data_path,soft_labels_path)
		if path is None:
			return None
		try:
			from LabGym.training.soft_labels import SoftLabelTable
			table=SoftLabelTable.load_csv(path)
			means={}
			buckets={c:[] for c in classnames}
			for hard,soft in table.rows.values():
				if hard in buckets:
					buckets[hard].append(soft)
			for c in classnames:
				vecs=buckets.get(c) or []
				if not vecs:
					continue
				m=np.mean(np.stack(vecs,axis=0),axis=0)
				# map table class order -> training class order
				aligned=np.zeros(len(classnames),dtype=np.float32)
				for ti,tn in enumerate(table.classnames):
					if tn in classnames:
						aligned[list(classnames).index(tn)]=m[ti]
				s=float(aligned.sum())
				if s>1e-8:
					aligned=aligned/s
				means[c]=aligned
			return means if means else None
		except Exception as exc:
			print('class mean soft failed: '+str(exc))
			return None


	def _apply_soft_to_batch_labels(self,hard_Y,classnames,class_means,label_mode):
		'''Build stacked [hard|soft] targets; soft from class means or hard copy.'''
		return apply_class_mean_soft_to_labels(hard_Y,classnames,class_means,label_mode)


	# Single-file artifact inside the categorizer folder. Avoids TensorFlow
	# SavedModel nested temp paths (…/variables/variables_temp/…) that often
	# exceed Windows MAX_PATH (~260) when model folder names are long.
	CATEGORIZER_MODEL_FILENAME='model.keras'

	@staticmethod
	def categorizer_model_file(model_path):
		'''Path to the primary Keras model file inside a categorizer folder.'''
		return os.path.join(model_path,Categorizers.CATEGORIZER_MODEL_FILENAME)

	@staticmethod
	def load_categorizer_model(model_path,compile=True):
		'''Load a categorizer from folder (prefer model.keras; fall back to SavedModel dir).'''
		keras_file=Categorizers.categorizer_model_file(model_path)
		if os.path.isfile(keras_file):
			return load_model(keras_file,compile=compile)
		# Legacy LabGym layout: whole folder is a SavedModel
		return load_model(model_path,compile=compile)

	@staticmethod
	def save_categorizer_model(model,model_path):
		'''Save categorizer as model.keras inside model_path (Windows-safe).'''
		os.makedirs(model_path,exist_ok=True)
		out=Categorizers.categorizer_model_file(model_path)
		model.save(out)
		return out

	def _standard_fit_callbacks(self,model_path,train_progress_cb=None,cancel_event=None):
		'''Checkpoint + early stop + LR plateau + optional epoch progress / cancel callbacks.'''
		from LabGym.training.progress import (
			make_cancel_callback,
			make_epoch_progress_callback,
		)

		os.makedirs(model_path,exist_ok=True)
		# Single-file .keras checkpoint: much shorter paths than SavedModel dirs on Windows
		ckpt_path=self.categorizer_model_file(model_path)
		cp=ModelCheckpoint(ckpt_path,monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')
		es=EarlyStopping(monitor='val_loss',min_delta=0.001,mode='min',verbose=1,patience=6,restore_best_weights=True)
		rl=ReduceLROnPlateau(monitor='val_loss',min_delta=0.001,factor=0.2,patience=3,verbose=1,mode='min',min_lr=1e-7)
		cbs=[cp,es,rl]
		ep=make_epoch_progress_callback(train_progress_cb)
		if ep is not None:
			cbs.append(ep)
		cc=make_cancel_callback(cancel_event)
		if cc is not None:
			cbs.append(cc)
		return cbs


	def _raise_if_fit_cancelled(self,cancel_event,phase='Training'):
		from LabGym.training.progress import is_cancelled,TrainingCancelled
		if is_cancelled(cancel_event):
			msg=phase+' cancelled by user.'
			print(msg)
			self.log.append(msg)
			raise TrainingCancelled(msg)


	@staticmethod
	def has_exported_aug_data(out_folder):
		'''True if out_folder has train/ and validation/ with at least one .jpg each.'''
		if not out_folder:
			return False
		train_folder=os.path.join(out_folder,'train')
		validation_folder=os.path.join(out_folder,'validation')
		if not (os.path.isdir(train_folder) and os.path.isdir(validation_folder)):
			return False
		def _has_jpg(folder):
			try:
				return any(name.endswith('.jpg') for name in os.listdir(folder))
			except OSError:
				return False
		return _has_jpg(train_folder) and _has_jpg(validation_folder)


	def rename_label(self,file_path,new_path,resize=None):

		# file_path: the folder that stores the sorted, unprepared examples
		# new_path: the folder that stores all prepared examples, which can be directly used for training a Categorizer
		# resize: if not None, resize the frames in animations / pattern images to the target size

		folder_list=[i for i in os.listdir(file_path) if os.path.isdir(os.path.join(file_path,i))]

		if len(folder_list)<2:

			print('You need at least 2 categories of behaviors!')
			print('Preparation aborted!')

		else:

			print('Behavior names are: '+str(folder_list))
			previous_lenth=None
			imagedata=False

			for folder in folder_list:

				name_list=[i for i in os.listdir(os.path.join(file_path,folder)) if i.endswith('.avi')]

				if len(name_list)==0:
					name_list=[i for i in os.listdir(os.path.join(file_path,folder)) if i.endswith('.jpg')]
					imagedata=True

				for i in name_list:

					if imagedata:

						image=os.path.join(file_path,folder,i)
						new_image=os.path.join(new_path,str(name_list.index(i))+'_'+folder+'.jpg')
						image=cv2.imread(image)
						if resize is not None:
							image=cv2.resize(image,(resize,resize),interpolation=cv2.INTER_AREA)
						cv2.imwrite(new_image,image)

					else:

						animation=os.path.join(file_path,folder,i)
						pattern_image=os.path.join(file_path,folder,os.path.splitext(i)[0]+'.jpg')
						current_length=0

						new_animation=os.path.join(new_path,str(name_list.index(i))+'_'+folder+'.avi')
						new_pattern_image=os.path.join(new_path,str(name_list.index(i))+'_'+folder+'.jpg')
						writer=None
						capture=cv2.VideoCapture(animation)
						fps=round(capture.get(cv2.CAP_PROP_FPS))
						while True:
							retval,frame=capture.read()
							current_length+=1
							if frame is None:
								break
							if resize is not None:
								frame=cv2.resize(frame,(resize,resize),interpolation=cv2.INTER_AREA)
							if writer is None:
								(h,w)=frame.shape[:2]
								writer=cv2.VideoWriter(new_animation,cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h),True)
							writer.write(frame)
						capture.release()
						writer.release()
						pattern_image=cv2.imread(pattern_image)
						if resize is not None:
							pattern_image=cv2.resize(pattern_image,(resize,resize),interpolation=cv2.INTER_AREA)
						cv2.imwrite(new_pattern_image,pattern_image)
						if previous_lenth is None:
							previous_lenth=current_length
						else:
							if previous_lenth!=current_length:
								previous_lenth=current_length
								print('Inconsistent duration of animation detected at: '+str(i)+'. Check the duration of animations!')

			print('All prepared training examples stored in: '+str(new_path))


	def build_data(self,path_to_animations,dim_tconv=0,dim_conv=64,channel=1,time_step=15,aug_methods=[],background_free=True,black_background=True,behavior_mode=0,out_path=None,num_workers=1,progress_cb=None,cancel_event=None):

		# path_to_animations: list of paths to prepared training examples (videos or images)
		# dim_tconv: the input dimension of Animation Analyzer
		# dim_conv: the input dimension of Pattern Recognizer
		# channel: the input color channel of Animation Analyzer, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation, also the input length of Animation Analyzer
		# aug_methods: the augmentation methods that are used in training
		# background_free: whether the background is included in animations
		# black_background: whether to set background black
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# out_path: if not None, will output all the augmented data to this path
		# num_workers: process-pool size for export path only (>1). In-memory stays sequential.
		# progress_cb: optional callable(done_sources, total_sources, message)
		# cancel_event: optional threading.Event / callable; cooperative cancel between sources

		from LabGym.training.progress import is_cancelled,raise_if_cancelled,TrainingCancelled

		animations=deque()
		pattern_images=deque()
		labels=deque()
		amount=0

		path_to_animations=list(path_to_animations or [])
		total_sources=len(path_to_animations)
		methods=resolve_aug_methods(aug_methods)
		try:
			num_workers=max(1,int(num_workers or 1))
		except (TypeError,ValueError):
			num_workers=1

		def _report(done_sources,msg=None):
			if progress_cb is None:
				return
			try:
				progress_cb(
					done_sources,
					total_sources,
					msg or ('Augmenting… %d/%d sources (%d outputs)'%(done_sources,total_sources,amount)),
				)
			except Exception:
				pass

		if total_sources==0:
			msg='No source examples to augment.'
			print(msg)
			self.log.append(msg)
			_report(0,msg)
			if out_path is None:
				if dim_tconv!=0:
					animations=np.array([],dtype='float32')
				pattern_images=np.array([],dtype='float32')
				labels=np.array([])
			return animations,pattern_images,labels

		# Parallel export only; in-memory stays sequential.
		use_pool=(
			out_path is not None
			and num_workers>1
			and total_sources>=16
		)
		workers=1
		if use_pool:
			workers=max(1,min(num_workers,total_sources))
			msg='Augmenting with %d workers (%d sources)…'%(workers,total_sources)
			print(msg)
			self.log.append(msg)

		raise_if_cancelled(cancel_event,'Augmentation cancelled by user')

		if use_pool and workers>1:
			payloads=[]
			for idx,path in enumerate(path_to_animations):
				payloads.append({
					'animation_path':path,
					'methods':methods,
					'dim_tconv':dim_tconv,
					'dim_conv':dim_conv,
					'channel':channel,
					'time_step':time_step,
					'background_free':background_free,
					'black_background':black_background,
					'behavior_mode':behavior_mode,
					'out_path':out_path,
					'seed':(idx+1)*10007,
				})
			done_sources=0
			_report(0)
			try:
				with ProcessPoolExecutor(
					max_workers=workers,
					initializer=init_augment_worker,
				) as pool:
					futures={pool.submit(augment_export_task,p):p for p in payloads}
					for fut in as_completed(futures):
						if is_cancelled(cancel_event):
							# Stop waiting; cancel pending (Python 3.9+)
							try:
								pool.shutdown(wait=False,cancel_futures=True)
							except TypeError:
								for f in futures:
									f.cancel()
							msg='Augmentation cancelled by user after %d/%d sources.'%(done_sources,total_sources)
							print(msg)
							self.log.append(msg)
							_report(done_sources,msg)
							raise TrainingCancelled(msg)
						anims,patterns,labs,n_done,warnings=fut.result()
						for w in warnings:
							print(w)
							self.log.append(w)
						amount+=n_done
						done_sources+=1
						if amount>0 and amount%10000<max(n_done,1):
							print('The augmented example amount: '+str(amount))
							self.log.append('The augmented example amount: '+str(amount))
							print(datetime.datetime.now())
							self.log.append(str(datetime.datetime.now()))
						_report(done_sources)
			except TrainingCancelled:
				raise
			except Exception as exc:
				print('Parallel augmentation failed (%s); falling back to sequential.'%exc)
				self.log.append('Parallel augmentation failed: '+str(exc)+'; using sequential.')
				# Sequential fallback for remaining would be complex; re-run all sequential
				# only if nothing written yet — otherwise re-raise.
				if done_sources==0:
					return self.build_data(
						path_to_animations,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,
						time_step=time_step,aug_methods=aug_methods,background_free=background_free,
						black_background=black_background,behavior_mode=behavior_mode,
						out_path=out_path,num_workers=1,progress_cb=progress_cb,cancel_event=cancel_event)
				raise
		else:
			for idx,i in enumerate(path_to_animations):
				raise_if_cancelled(cancel_event,'Augmentation cancelled by user')
				anims,patterns,labs,n_done,warnings=augment_one_example(
					i,
					methods,
					dim_tconv=dim_tconv,
					dim_conv=dim_conv,
					channel=channel,
					time_step=time_step,
					background_free=background_free,
					black_background=black_background,
					behavior_mode=behavior_mode,
					out_path=out_path,
					seed=None,
				)
				for w in warnings:
					print(w)
					self.log.append(w)

				if out_path is None:
					if dim_tconv!=0 and anims:
						animations.extend(anims)
					if patterns:
						pattern_images.extend(patterns)
					if labs:
						labels.extend(labs)
				amount+=n_done
				if amount>0 and amount%10000==0:
					print('The augmented example amount: '+str(amount))
					self.log.append('The augmented example amount: '+str(amount))
					print(datetime.datetime.now())
					self.log.append(str(datetime.datetime.now()))
				_report(idx+1)

		if out_path is None:

			if dim_tconv!=0:
				animations=np.array(animations,dtype='float32')/255.0
			pattern_images=np.array(pattern_images,dtype='float32')/255.0
			labels=np.array(labels)

		_report(total_sources,'Augmentation complete (%d outputs).'%amount)
		return animations,pattern_images,labels


	def simple_vgg(self,inputs,filters,classes=3,level=2,with_classifier=False):

		# inputs: the input tensor (w,h,c) of the neural network
		# filters: the number of nodes (neurons) in each layer
		# classes: the behavior category names (if with_classifier is True)
		# level: complexity level, determines how deep the neural network is
		# with_classifier: if True, the neural network can output classification probabilities

		if level<2:
			layers=[2]
		elif level<3:
			layers=[2,3]
		elif level<4:
			layers=[2,3,4]
		else:
			layers=[2,3,4,4]

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(inputs)
						x=BatchNormalization()(x)
					else:
						x=Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(x)
						x=BatchNormalization()(x)
				else:
					x=Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(x)
					x=BatchNormalization()(x)
			x=MaxPooling2D(pool_size=(2,2))(x)
			filters=int(filters*2)

		x=Flatten()(x)

		if with_classifier is False:

			return x

		else:

			x=Dense(int(filters/2),activation='relu')(x)
			x=BatchNormalization()(x)
			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)
			#plot_model(model,'model.png',show_shapes=True)

			return model


	def simple_tvgg(self,inputs,filters,classes=3,level=2,with_classifier=False):

		# inputs: the input tensor (t,w,h,c) of the neural network
		# filters: the number of nodes (neurons) in each layer
		# classes: the behavior category names (if with_classifier is True)
		# level: complexity level, determines how deep the neural network is
		# with_classifier: if True, the neural network can output classification probabilities

		if level<2:
			layers=[2]
		elif level<3:
			layers=[2,3]
		elif level<4:
			layers=[2,3,4]
		else:
			layers=[2,3,4,4]

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=TimeDistributed(Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu'))(inputs)
						x=TimeDistributed(BatchNormalization())(x)
					else:
						x=TimeDistributed(Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu'))(x)
						x=TimeDistributed(BatchNormalization())(x)
				else:
					x=TimeDistributed(Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu'))(x)
					x=TimeDistributed(BatchNormalization())(x)
			x=TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
			filters=int(filters*2)

		x=TimeDistributed(Flatten())(x)
		x=LSTM(int(filters/2),return_sequences=False,return_state=False)(x)

		if with_classifier is False:

			return x

		else:

			x=Dense(int(filters/2),activation='relu')(x)
			x=BatchNormalization()(x)
			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)
			#plot_model(model,'model.png',show_shapes=True)

			return model


	def res_block(self,x,filters,strides=2,block=False,basic=True):

		# x: the output from the last layer
		# filters: the number of nodes (neurons) in each layer
		# strides: the strides in each layer
		# block: whether it's a block or shortcut
		# basic: whether it uses additional zeropadding and normalization

		shortcut=x

		if basic:

			x=ZeroPadding2D((1,1))(x)
			x=Conv2D(filters,(3,3),strides=(strides,strides))(x)

		else:

			x=Conv2D(filters,(1,1),strides=(strides,strides))(x)

		x=BatchNormalization()(x)
		x=Activation('relu')(x)

		x=ZeroPadding2D((1,1))(x)
		x=Conv2D(filters,(3,3),strides=(1,1))(x)
		x=BatchNormalization()(x)

		if basic:

			if block is False:
				shortcut=Conv2D(filters,(1,1),strides=(strides,strides))(shortcut)
				shortcut=BatchNormalization()(shortcut)

		else:

			x=Activation('relu')(x)

			x=Conv2D(int(filters*4),(1,1),strides=(1,1))(x)
			x=BatchNormalization()(x)

			if block is False:
				shortcut=Conv2D(filters*4,(1,1),strides=(strides,strides))(shortcut)
				shortcut=BatchNormalization()(shortcut)

		x=Add()([x,shortcut])
		x=Activation('relu')(x)

		return x


	def tres_block(self,x,filters,strides=2,block=False,basic=True):

		# x: the output from the last layer
		# filters: the number of nodes (neurons) in each layer
		# strides: the strides in each layer
		# block: whether it's a block or shortcut
		# basic: whether it uses additional zeropadding and normalization

		shortcut=x

		if basic:

			x=TimeDistributed(ZeroPadding2D((1,1)))(x)
			x=TimeDistributed(Conv2D(filters,(3,3),strides=(strides,strides)))(x)

		else:

			x=TimeDistributed(Conv2D(filters,(1,1),strides=(strides,strides)))(x)

		x=TimeDistributed(BatchNormalization())(x)
		x=TimeDistributed(Activation('relu'))(x)

		x=TimeDistributed(ZeroPadding2D((1,1)))(x)
		x=TimeDistributed(Conv2D(filters,(3,3),strides=(1,1)))(x)
		x=TimeDistributed(BatchNormalization())(x)

		if basic:

			if block is False:
				shortcut=TimeDistributed(Conv2D(filters,(1,1),strides=(strides,strides)))(shortcut)
				shortcut=TimeDistributed(BatchNormalization())(shortcut)

		else:

			x=TimeDistributed(Activation('relu'))(x)

			x=TimeDistributed(Conv2D(int(filters*4),(1,1),strides=(1,1)))(x)
			x=TimeDistributed(BatchNormalization())(x)

			if block is False:
				shortcut=TimeDistributed(Conv2D(int(filters*4),(1,1),strides=(strides,strides)))(shortcut)
				shortcut=TimeDistributed(BatchNormalization())(shortcut)

		x=Add()([x,shortcut])
		x=TimeDistributed(Activation('relu'))(x)

		return x


	def simple_resnet(self,inputs,filters,classes=3,level=5,with_classifier=False):

		# inputs: the input tensor (w,h,c) of the neural network
		# filters: the number of nodes (neurons) in each layer
		# classes: the behavior category names (if with_classifier is True)
		# level: complexity level, determines how deep the neural network is
		# with_classifier: if True, the neural network can output classification probabilities

		x=ZeroPadding2D((3,3))(inputs)
		x=Conv2D(filters,(5,5),strides=(2,2))(x)
		x=BatchNormalization()(x)
		x=Activation('relu')(x)
		x=MaxPooling2D((3,3),strides=(2,2))(x)

		if level<6:
			layers=[2,2,2,2]
			basic=True
		elif level<7:
			layers=[3,4,6,3]
			basic=True
		else:
			layers=[3,4,6,3]
			basic=False

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=self.res_block(x,filters,strides=1,block=False,basic=basic)
					else:
						x=self.res_block(x,filters,strides=2,block=False,basic=basic)
				else:
					x=self.res_block(x,filters,strides=1,block=True,basic=basic)
			filters=int(filters*2)

		x=AveragePooling2D((2,2))(x)
		x=Flatten()(x)

		if with_classifier is False:

			return x

		else:

			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)

			return model


	def simple_tresnet(self,inputs,filters,classes=3,level=5,with_classifier=False):

		# inputs: the input tensor (t,w,h,c) of the neural network
		# filters: the number of nodes (neurons) in each layer
		# classes: the behavior category names (if with_classifier is True)
		# level: complexity level, determines how deep the neural network is
		# with_classifier: if True, the neural network can output classification probabilities

		x=TimeDistributed(ZeroPadding2D((3,3)))(inputs)
		x=TimeDistributed(Conv2D(filters,(5,5),strides=(2,2)))(x)
		x=TimeDistributed(BatchNormalization())(x)
		x=TimeDistributed(Activation('relu'))(x)
		x=TimeDistributed(MaxPooling2D((3,3),strides=(2,2)))(x)

		if level<6:
			layers=[2,2,2,2]
			basic=True
		elif level<7:
			layers=[3,4,6,3]
			basic=True
		else:
			layers=[3,4,6,3]
			basic=False

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=self.tres_block(x,filters,strides=1,block=False,basic=basic)
					else:
						x=self.tres_block(x,filters,strides=2,block=False,basic=basic)
				else:
					x=self.tres_block(x,filters,strides=1,block=True,basic=basic)
			filters=int(filters*2)

		x=TimeDistributed(AveragePooling2D((2,2)))(x)
		x=TimeDistributed(Flatten())(x)

		if level==5:
			x=LSTM(1024,return_sequences=False,return_state=False)(x)
		elif level==6:
			x=LSTM(2048,return_sequences=False,return_state=False)(x)
		else:
			x=LSTM(4096,return_sequences=False,return_state=False)(x)

		if with_classifier is False:

			return x

		else:

			if level==5:
				x=Dense(1024,activation='relu')(x)
			elif level==6:
				x=Dense(2048,activation='relu')(x)
			else:
				x=Dense(4096,activation='relu')(x)

			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)

			return model


	def combined_network(self,time_step=15,dim_tconv=32,dim_conv=64,channel=1,classes=9,level_tconv=1,level_conv=2):

		# time_step: the duration of an animation, also the input length of Animation Analyzer
		# dim_tconv: the input dimension of Animation Analyzer
		# dim_conv: the input dimension of Pattern Recognizer
		# channel: the input color channel of Animation Analyzer, 1 is gray scale, 3 is RGB
		# classes: the behavior category names
		# level_tconv: complexity level of Animation Analyzer, determines how deep the neural network is
		# level_conv: complexity level of Pattern Recognizer, determines how deep the neural network is

		animation_inputs=Input(shape=(time_step,dim_tconv,dim_tconv,channel))
		pattern_image_inputs=Input(shape=(dim_conv,dim_conv,3))

		filters_tconv=8
		filters_conv=8

		for i in range(round(dim_tconv/60)):
			filters_tconv=min(int(filters_tconv*2),64)

		for i in range(round(dim_conv/60)):
			filters_conv=min(int(filters_conv*2),64)

		if level_tconv<5:
			animation_feature=self.simple_tvgg(animation_inputs,filters_tconv,level=level_tconv,with_classifier=False)
		else:
			animation_feature=self.simple_tresnet(animation_inputs,filters_tconv,level=level_tconv,with_classifier=False)

		if level_conv<5:
			pattern_image_feature=self.simple_vgg(pattern_image_inputs,filters_conv,level=level_conv,with_classifier=False)
		else:
			pattern_image_feature=self.simple_resnet(pattern_image_inputs,filters_conv,level=level_conv,with_classifier=False)

		merged_features=concatenate([animation_feature,pattern_image_feature])

		nodes=32
		for i in range(max(level_tconv,level_conv)):
			nodes=int(nodes*2)
		outputs=Dense(nodes,activation='relu')(merged_features)
		outputs=BatchNormalization()(outputs)
		outputs=Dropout(0.5)(outputs)
		if classes==2:
			predictions=Dense(1,activation='sigmoid')(outputs)
		else:
			predictions=Dense(classes,activation='softmax')(outputs)

		model=Model(inputs=[animation_inputs,pattern_image_inputs],outputs=predictions)

		return model


	def train_pattern_recognizer(self,data_path,model_path,out_path=None,dim=64,channel=3,time_step=15,level=2,aug_methods=[],augvalid=True,include_bodyparts=True,std=0,background_free=True,black_background=True,behavior_mode=0,social_distance=0,out_folder=None,label_mode='hard_only',lambda_soft=0.4,soft_labels_path=None,num_workers=1,progress_cb=None,train_progress_cb=None,cancel_event=None,skip_augment=False):

		# data_path: the folder that stores all the prepared training examples
		# model_path: the path to the trained Pattern Recognizer
		# out_path: if not None, will store the training reports in this folder
		# dim: the input dimension
		# channel: the input color channel, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation / pattern image, also the length of a behavior episode
		# level: complexity level, determines how deep the neural network is
		# aug_methods: the augmentation methods that are used in training
		# augvalid: whether augment the validation data as well
		# include_bodyparts: whether to include body parts in the pattern images
		# std: a value between 0 and 255, higher value, less body parts will be included in the pattern images
		# background_free: whether to include background in animations
		# black_background: whether to set background
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# social_distance: a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		# out_folder: if not None, will output all the augmented data to this folder
		# num_workers: process workers for export augmentation (1 = sequential)
		# progress_cb: optional callable(done, total, message) for export augmentation
		# train_progress_cb: optional callable(epoch_1based, logs_dict) each training epoch

		filters=8

		for i in range(round(dim/60)):
			filters=min(int(filters*2),64)

		inputs=Input(shape=(dim,dim,channel))

		print('Training the Categorizer w/ only Pattern Recognizer using the behavior examples in: '+str(data_path))
		self.log.append('Training the Categorizer w/ only Pattern Recognizer using the behavior examples in: '+str(data_path))

		files=[i for i in os.listdir(data_path) if i.endswith(self.extension_image)]

		path_files=[]
		labels=[]

		for i in files:
			path_file=os.path.join(data_path,i)
			path_files.append(path_file)
			labels.append(os.path.splitext(i)[0].split('_')[-1])

		labels=np.array(labels)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		self.classnames=lb.classes_

		if len(list(self.classnames))<2:

			print('You need at least 2 categories of behaviors!')
			print('Training aborted!')

		else:

			print('Found behavior names: '+str(self.classnames))
			self.log.append('Found behavior names: '+str(self.classnames))

			if out_folder is None:

				if include_bodyparts:
					inner_code=0
				else:
					inner_code=1

				if background_free:
					background_code=0
				else:
					background_code=1

				if black_background:
					black_code=0
				else:
					black_code=1

				if behavior_mode>=3:
					time_step=std=0
					inner_code=1

				parameters={'classnames':list(self.classnames),'dim_conv':int(dim),'channel':int(channel),'time_step':int(time_step),'network':0,'level_conv':int(level),'inner_code':int(inner_code),'std':int(std),'background_free':int(background_code),'black_background':int(black_code),'behavior_kind':int(behavior_mode),'social_distance':int(social_distance),'label_mode':str(label_mode),'lambda_soft':float(lambda_soft)}
				pd_parameters=pd.DataFrame.from_dict(parameters)
				pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

				(train_files,test_files,y1,y2)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)
				self.label_mode=label_mode
				self.lambda_soft=lambda_soft
				class_means=None
				if str(label_mode)!='hard_only':
					class_means=self._class_mean_soft(data_path,list(self.classnames),soft_labels_path)
					if class_means is None:
						print('Soft labels not found; falling back to hard_only.')
						self.log.append('Soft labels not found; falling back to hard_only.')
						label_mode='hard_only'
					else:
						print('Using soft-label mode: '+str(label_mode)+' (lambda_soft='+str(lambda_soft)+')')
						self.log.append('Using soft-label mode: '+str(label_mode))

				print('Perform augmentation for the behavior examples...')
				self.log.append('Perform augmentation for the behavior examples...')
				print('This might take hours or days, depending on the capacity of your computer.')
				print(datetime.datetime.now())
				self.log.append(str(datetime.datetime.now()))

				print('Start to augment training examples...')
				self.log.append('Start to augment training examples...')
				_,trainX,trainY=self.build_data(train_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				trainY=lb.fit_transform(trainY)
				print('Start to augment validation examples...')
				self.log.append('Start to augment validation examples...')
				if augvalid:
					_,testX,testY=self.build_data(test_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				else:
					_,testX,testY=self.build_data(test_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				testY=lb.fit_transform(testY)
				trainY=self._apply_soft_to_batch_labels(trainY,list(self.classnames),class_means,label_mode)
				testY=self._apply_soft_to_batch_labels(testY,list(self.classnames),class_means,label_mode)

				with tf.device('CPU'):
					trainX=tf.convert_to_tensor(trainX)
					trainY=tf.convert_to_tensor(trainY)
					testX_tensor=tf.convert_to_tensor(testX)
					testY_tensor=tf.convert_to_tensor(testY)

				print('Training example shape : '+str(trainX.shape))
				self.log.append('Training example shape : '+str(trainX.shape))
				print('Training label shape : '+str(trainY.shape))
				self.log.append('Training label shape : '+str(trainY.shape))
				print('Validation example shape : '+str(testX.shape))
				self.log.append('Validation example shape : '+str(testX.shape))
				print('Validation label shape : '+str(testY.shape))
				self.log.append('Validation label shape : '+str(testY.shape))
				print(datetime.datetime.now())
				self.log.append(str(datetime.datetime.now()))

				if dim<=128:
					batch_size=32
				elif dim<=256:
					batch_size=16
				else:
					batch_size=8

				if level<5:
					model=self.simple_vgg(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
				else:
					model=self.simple_resnet(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
				self._compile_model(model,label_mode=label_mode,lambda_soft=lambda_soft)

				# validation tensors may be hard-only in older paths; ensure soft stack
				testY_tensor=testY
				H=model.fit(trainX,trainY,batch_size=batch_size,validation_data=(testX_tensor,testY_tensor),epochs=1000000,callbacks=self._standard_fit_callbacks(model_path,train_progress_cb,cancel_event))
				self._raise_if_fit_cancelled(cancel_event)

				self.save_categorizer_model(model,model_path)
				print('Trained Categorizer saved in: '+str(model_path))
				self.log.append('Trained Categorizer saved in: '+str(model_path))

				predictions=model.predict(testX,batch_size=batch_size)

				# Metrics use hard half of stacked targets when present
				testY_hard=np.asarray(testY)
				C=len(self.classnames)
				if testY_hard.ndim==2 and testY_hard.shape[1]==2*C:
					testY_hard=testY_hard[:,:C]
				elif testY_hard.ndim==2 and C==2 and testY_hard.shape[1]==2:
					testY_hard=testY_hard[:,:1]

				if len(self.classnames)==2:
					predictions=[round(i[0]) for i in predictions]
					print(classification_report(testY_hard,predictions,target_names=self.classnames))
					report=classification_report(testY_hard,predictions,target_names=self.classnames,output_dict=True)
				else:
					print(classification_report(testY_hard.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames))
					report=classification_report(testY_hard.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames,output_dict=True)

				pd.DataFrame(report).transpose().to_csv(os.path.join(model_path,'training_metrics.csv'),float_format='%.2f')
				if out_path is not None:
					pd.DataFrame(report).transpose().to_excel(os.path.join(out_path,'training_metrics.xlsx'),float_format='%.2f')

				plt.style.use('classic')
				plt.figure()
				plt.plot(H.history['loss'],label='train_loss')
				plt.plot(H.history['val_loss'],label='val_loss')
				plt.plot(H.history['accuracy'],label='train_accuracy')
				plt.plot(H.history['val_accuracy'],label='val_accuracy')
				plt.title('Loss and Accuracy')
				plt.xlabel('Epoch')
				plt.ylabel('Loss/Accuracy')
				plt.legend(loc='center right')
				plt.savefig(os.path.join(model_path,'training_history.png'))
				if out_path is not None:
					plt.savefig(os.path.join(out_path,'training_history.png'))
					print('Training reports saved in: '+str(out_path))
					if len(self.log)>0:
						with open(os.path.join(out_path,'Training log.txt'),'w') as training_log:
							training_log.write('\n'.join(str(i) for i in self.log))
				plt.close('all')

			else:

				(train_files,test_files,_,_)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

				self.label_mode=label_mode
				self.lambda_soft=lambda_soft
				if str(label_mode)!='hard_only':
					class_means=self._class_mean_soft(data_path,list(self.classnames),soft_labels_path)
					if class_means is None:
						print('Soft labels not found; falling back to hard_only.')
						self.log.append('Soft labels not found; falling back to hard_only.')
						label_mode='hard_only'
						self.label_mode=label_mode
					else:
						print('Using soft-label mode: '+str(label_mode)+' (lambda_soft='+str(lambda_soft)+')')
						self.log.append('Using soft-label mode: '+str(label_mode))

				reuse=bool(skip_augment) and self.has_exported_aug_data(out_folder)
				if reuse:
					msg='Reusing existing augmented export (skip re-augment): '+str(out_folder)
					print(msg)
					self.log.append(msg)
					if progress_cb is not None:
						try:
							progress_cb(1,1,'Reusing existing augmented data…')
						except Exception:
							pass
				else:
					if skip_augment:
						msg='skip_augment requested but export incomplete; re-augmenting to: '+str(out_folder)
						print(msg)
						self.log.append(msg)
					print('Perform augmentation for the behavior examples and export them to: '+str(out_folder))
					self.log.append('Perform augmentation for the behavior examples and export them to: '+str(out_folder))
					print('This might take hours or days, depending on the capacity of your computer.')
					print(datetime.datetime.now())
					self.log.append(str(datetime.datetime.now()))

					print('Start to augment training examples...')
					self.log.append('Start to augment training examples...')
					train_folder=os.path.join(out_folder,'train')
					os.makedirs(train_folder,exist_ok=True)
					n_train=len(train_files)
					n_val=len(test_files)
					aug_total=n_train+n_val
					def _phase_cb(offset):
						if progress_cb is None:
							return None
						def _cb(done,tot,msg):
							progress_cb(offset+done,aug_total,msg)
						return _cb
					_,_,_=self.build_data(train_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=train_folder,num_workers=num_workers,progress_cb=_phase_cb(0),cancel_event=cancel_event)
					print('Start to augment validation examples...')
					self.log.append('Start to augment validation examples...')
					validation_folder=os.path.join(out_folder,'validation')
					os.makedirs(validation_folder,exist_ok=True)
					if augvalid:
						_,_,_=self.build_data(test_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=validation_folder,num_workers=num_workers,progress_cb=_phase_cb(n_train),cancel_event=cancel_event)
					else:
						_,_,_=self.build_data(test_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=validation_folder,num_workers=num_workers,progress_cb=_phase_cb(n_train),cancel_event=cancel_event)

				self.train_pattern_recognizer_onfly(
					out_folder,model_path,out_path=out_path,dim=dim,channel=channel,time_step=time_step,level=level,
					include_bodyparts=include_bodyparts,std=std,background_free=background_free,
					black_background=black_background,behavior_mode=behavior_mode,social_distance=social_distance,
					label_mode=label_mode,lambda_soft=lambda_soft,soft_labels_path=soft_labels_path,
					soft_source_path=data_path,train_progress_cb=train_progress_cb,cancel_event=cancel_event)


	def train_animation_analyzer(self,data_path,model_path,out_path=None,dim=64,channel=1,time_step=15,level=2,aug_methods=[],augvalid=True,include_bodyparts=True,std=0,background_free=True,black_background=True,behavior_mode=0,social_distance=0,color_costar=False,out_folder=None,num_workers=1,progress_cb=None,train_progress_cb=None,cancel_event=None,skip_augment=False):

		# data_path: the folder that stores all the prepared training examples
		# model_path: the path to the trained Animation Analyzer
		# out_path: if not None, will store the training reports in this folder
		# dim: the input dimension
		# channel: the input color channel, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation, also the input length of Animation Analyzer
		# level: complexity level, determines how deep the neural network is
		# aug_methods: the augmentation methods that are used in training
		# augvalid: whether augment the validation data as well
		# include_bodyparts: whether to include body parts in the pattern images
		# std: a value between 0 and 255, higher value, less body parts will be included in the pattern images
		# background_free: whether to include background in animations
		# black_background: whether to set background
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# social_distance: a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		# color_costar: in 'interactive advanced' mode, whether to make the supporting roles RGB scale in animations
		# out_folder: if not None, will output all the augmented data to this folder

		filters=8

		for i in range(round(dim/60)):
			filters=min(int(filters*2),64)

		inputs=Input(shape=(time_step,dim,dim,channel))

		print('Training the Categorizer w/o Pattern Recognizer using the behavior examples in: '+str(data_path))
		self.log.append('Training the Categorizer w/ only Pattern Recognizer using the behavior examples in: '+str(data_path))

		files=[i for i in os.listdir(data_path) if i.endswith(self.extension_video)]

		path_files=[]
		labels=[]

		for i in files:
			path_file=os.path.join(data_path,i)
			path_files.append(path_file)
			labels.append(os.path.splitext(i)[0].split('_')[-1])

		labels=np.array(labels)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		self.classnames=lb.classes_

		if len(list(self.classnames))<2:

			print('You need at least 2 categories of behaviors!')
			print('Training aborted!')

		else:

			print('Found behavior names: '+str(self.classnames))
			self.log.append('Found behavior names: '+str(self.classnames))

			if out_folder is None:

				if include_bodyparts:
					inner_code=0
				else:
					inner_code=1

				if background_free:
					background_code=0
				else:
					background_code=1

				if black_background:
					black_code=0
				else:
					black_code=1

				if color_costar:
					color_code=0
				else:
					color_code=1

				parameters={'classnames':list(self.classnames),'dim_tconv':int(dim),'channel':int(channel),'time_step':int(time_step),'network':1,'level_tconv':int(level),'inner_code':int(inner_code),'std':int(std),'background_free':int(background_code),'black_background':int(black_code),'behavior_kind':int(behavior_mode),'social_distance':int(social_distance),'color_code':int(color_code)}
				pd_parameters=pd.DataFrame.from_dict(parameters)
				pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

				(train_files,test_files,y1,y2)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

				print('Perform augmentation for the behavior examples...')
				self.log.append('Perform augmentation for the behavior examples...')
				print('This might take hours or days, depending on the capacity of your computer.')
				print(datetime.datetime.now())
				self.log.append(str(datetime.datetime.now()))

				print('Start to augment training examples...')
				self.log.append('Start to augment training examples...')
				trainX,_,trainY=self.build_data(train_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				trainY=lb.fit_transform(trainY)
				print('Start to augment validation examples...')
				self.log.append('Start to augment validation examples...')
				if augvalid:
					testX,_,testY=self.build_data(test_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				else:
					testX,_,testY=self.build_data(test_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				testY=lb.fit_transform(testY)

				with tf.device('CPU'):
					trainX=tf.convert_to_tensor(trainX)
					trainY=tf.convert_to_tensor(trainY)
					testX_tensor=tf.convert_to_tensor(testX)
					testY_tensor=tf.convert_to_tensor(testY)

				print('Training example shape : '+str(trainX.shape))
				self.log.append('Training example shape : '+str(trainX.shape))
				print('Training label shape : '+str(trainY.shape))
				self.log.append('Training label shape : '+str(trainY.shape))
				print('Validation example shape : '+str(testX.shape))
				self.log.append('Validation example shape : '+str(testX.shape))
				print('Validation label shape : '+str(testY.shape))
				self.log.append('Validation label shape : '+str(testY.shape))
				print(datetime.datetime.now())
				self.log.append(str(datetime.datetime.now()))

				if dim<=16:
					batch_size=32
				elif dim<=64:
					batch_size=16
				elif dim<=128:
					batch_size=8
				else:
					batch_size=4

				if level<5:
					model=self.simple_tvgg(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
				else:
					model=self.simple_tresnet(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)

				self._compile_model(model,label_mode=getattr(self,'label_mode','hard_only'),lambda_soft=getattr(self,'lambda_soft',0.4))

				H=model.fit(trainX,trainY,batch_size=batch_size,validation_data=(testX_tensor,testY_tensor),epochs=1000000,callbacks=self._standard_fit_callbacks(model_path,train_progress_cb,cancel_event))
				self._raise_if_fit_cancelled(cancel_event)

				self.save_categorizer_model(model,model_path)
				print('Trained Categorizer saved in: '+str(model_path))
				self.log.append('Trained Categorizer saved in: '+str(model_path))

				predictions=model.predict(testX,batch_size=batch_size)

				if len(self.classnames)==2:
					predictions=[round(i[0]) for i in predictions]
					print(classification_report(testY,predictions,target_names=self.classnames))
					report=classification_report(testY,predictions,target_names=self.classnames,output_dict=True)
				else:
					print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames))
					report=classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames,output_dict=True)

				pd.DataFrame(report).transpose().to_csv(os.path.join(model_path,'training_metrics.csv'),float_format='%.2f')
				if out_path is not None:
					pd.DataFrame(report).transpose().to_excel(os.path.join(out_path,'training_metrics.xlsx'),float_format='%.2f')

				plt.style.use('classic')
				plt.figure()
				plt.plot(H.history['loss'],label='train_loss')
				plt.plot(H.history['val_loss'],label='val_loss')
				plt.plot(H.history['accuracy'],label='train_accuracy')
				plt.plot(H.history['val_accuracy'],label='val_accuracy')
				plt.title('Loss and Accuracy')
				plt.xlabel('Epoch')
				plt.ylabel('Loss/Accuracy')
				plt.legend(loc='center right')
				plt.savefig(os.path.join(model_path,'training_history.png'))
				if out_path is not None:
					plt.savefig(os.path.join(out_path,'training_history.png'))
					print('Training reports saved in: '+str(out_path))
					if len(self.log)>0:
						with open(os.path.join(out_path,'Training log.txt'),'w') as training_log:
							training_log.write('\n'.join(str(i) for i in self.log))
				plt.close('all')

			else:

				(train_files,test_files,_,_)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

				reuse=bool(skip_augment) and self.has_exported_aug_data(out_folder)
				if reuse:
					msg='Reusing existing augmented export (skip re-augment): '+str(out_folder)
					print(msg)
					self.log.append(msg)
					if progress_cb is not None:
						try:
							progress_cb(1,1,'Reusing existing augmented data…')
						except Exception:
							pass
				else:
					if skip_augment:
						msg='skip_augment requested but export incomplete; re-augmenting to: '+str(out_folder)
						print(msg)
						self.log.append(msg)
					print('Perform augmentation for the behavior examples and export them to: '+str(out_folder))
					self.log.append('Perform augmentation for the behavior examples and export them to: '+str(out_folder))
					print('This might take hours or days, depending on the capacity of your computer.')
					print(datetime.datetime.now())
					self.log.append(str(datetime.datetime.now()))

					print('Start to augment training examples...')
					self.log.append('Start to augment training examples...')
					train_folder=os.path.join(out_folder,'train')
					os.makedirs(train_folder,exist_ok=True)
					n_train=len(train_files)
					n_val=len(test_files)
					aug_total=n_train+n_val
					def _phase_cb(offset):
						if progress_cb is None:
							return None
						def _cb(done,tot,msg):
							progress_cb(offset+done,aug_total,msg)
						return _cb
					_,_,_=self.build_data(train_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=train_folder,num_workers=num_workers,progress_cb=_phase_cb(0),cancel_event=cancel_event)
					print('Start to augment validation examples...')
					self.log.append('Start to augment validation examples...')
					validation_folder=os.path.join(out_folder,'validation')
					os.makedirs(validation_folder,exist_ok=True)
					if augvalid:
						_,_,_=self.build_data(test_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=validation_folder,num_workers=num_workers,progress_cb=_phase_cb(n_train),cancel_event=cancel_event)
					else:
						_,_,_=self.build_data(test_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=validation_folder,num_workers=num_workers,progress_cb=_phase_cb(n_train),cancel_event=cancel_event)

				self.train_animation_analyzer_onfly(out_folder,model_path,out_path=out_path,dim=dim,channel=channel,time_step=time_step,level=level,include_bodyparts=include_bodyparts,std=std,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,social_distance=social_distance,train_progress_cb=train_progress_cb,cancel_event=cancel_event)


	def train_combnet(self,data_path,model_path,out_path=None,dim_tconv=32,dim_conv=64,channel=1,time_step=15,level_tconv=1,level_conv=2,aug_methods=[],augvalid=True,include_bodyparts=True,std=0,background_free=True,black_background=True,behavior_mode=0,social_distance=0,color_costar=False,out_folder=None,label_mode='hard_only',lambda_soft=0.4,soft_labels_path=None,num_workers=1,progress_cb=None,train_progress_cb=None,cancel_event=None,skip_augment=False):

		# data_path: the folder that stores all the prepared training examples
		# model_path: the path to the trained Categorizer
		# out_path: if not None, will store the training reports in this folder
		# dim_tconv: the input dimension of Animation Analyzer
		# dim_conv: the input dimension of Pattern Recognizer
		# channel: the input color channel of Animation Analyzer, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation, also the input length of Animation Analyzer
		# level_tconv: complexity level of Animation Analyzer, determines how deep the neural network is
		# level_conv: complexity level of Pattern Recognizer, determines how deep the neural network is
		# aug_methods: the augmentation methods that are used in training
		# augvalid: whether augment the validation data as well
		# include_bodyparts: whether to include body parts in the pattern images
		# std: a value between 0 and 255, higher value, less body parts will be included in the pattern images
		# background_free: whether to include background in animations
		# black_background: whether to set background black
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# social_distance: a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		# color_costar: in 'interactive advanced' mode, whether to make the supporting roles RGB scale in animations
		# out_folder: if not None, will output all the augmented data to this folder
		# num_workers: process workers for export augmentation (1 = sequential)
		# progress_cb: optional callable(done, total, message) for export augmentation

		print('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))
		self.log.append('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))

		files=[i for i in os.listdir(data_path) if i.endswith(self.extension_video)]

		path_files=[]
		labels=[]

		for i in files:
			path_file=os.path.join(data_path,i)
			path_files.append(path_file)
			labels.append(os.path.splitext(i)[0].split('_')[-1])

		labels=np.array(labels)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		self.classnames=lb.classes_

		if len(list(self.classnames))<2:

			print('You need at least 2 categories of behaviors!')
			print('Training aborted!')

		else:

			print('Found behavior names: '+str(self.classnames))
			self.log.append('Found behavior names: '+str(self.classnames))

			if out_folder is None:

				if include_bodyparts:
					inner_code=0
				else:
					inner_code=1

				if background_free:
					background_code=0
				else:
					background_code=1

				if black_background:
					black_code=0
				else:
					black_code=1

				if color_costar:
					color_code=0
				else:
					color_code=1

				parameters={'classnames':list(self.classnames),'dim_tconv':int(dim_tconv),'dim_conv':int(dim_conv),'channel':int(channel),'time_step':int(time_step),'network':2,'level_tconv':int(level_tconv),'level_conv':int(level_conv),'inner_code':int(inner_code),'std':int(std),'background_free':int(background_code),'black_background':int(black_code),'behavior_kind':int(behavior_mode),'social_distance':int(social_distance),'color_code':int(color_code),'label_mode':str(label_mode),'lambda_soft':float(lambda_soft)}
				pd_parameters=pd.DataFrame.from_dict(parameters)
				pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

				(train_files,test_files,y1,y2)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)
				self.label_mode=label_mode
				self.lambda_soft=lambda_soft
				class_means=None
				if str(label_mode)!='hard_only':
					class_means=self._class_mean_soft(data_path,list(self.classnames),soft_labels_path)
					if class_means is None:
						print('Soft labels not found; falling back to hard_only.')
						self.log.append('Soft labels not found; falling back to hard_only.')
						label_mode='hard_only'
						self.label_mode=label_mode
					else:
						print('Using soft-label mode: '+str(label_mode)+' (lambda_soft='+str(lambda_soft)+')')
						self.log.append('Using soft-label mode: '+str(label_mode))

				print('Perform augmentation for the behavior examples...')
				self.log.append('Perform augmentation for the behavior examples...')
				print('This might take hours or days, depending on the capacity of your computer.')
				print(datetime.datetime.now())
				self.log.append(str(datetime.datetime.now()))

				print('Start to augment training examples...')
				self.log.append('Start to augment training examples...')
				train_animations,train_pattern_images,trainY=self.build_data(train_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				trainY=lb.fit_transform(trainY)
				print('Start to augment validation examples...')
				self.log.append('Start to augment validation examples...')
				if augvalid:
					test_animations,test_pattern_images,testY=self.build_data(test_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				else:
					test_animations,test_pattern_images,testY=self.build_data(test_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,cancel_event=cancel_event)
				testY=lb.fit_transform(testY)
				trainY=self._apply_soft_to_batch_labels(trainY,list(self.classnames),class_means,label_mode)
				testY=self._apply_soft_to_batch_labels(testY,list(self.classnames),class_means,label_mode)

				with tf.device('CPU'):
					train_animations=tf.convert_to_tensor(train_animations)
					train_pattern_images=tf.convert_to_tensor(train_pattern_images)
					trainY=tf.convert_to_tensor(trainY)
					test_animations_tensor=tf.convert_to_tensor(test_animations)
					test_pattern_images_tensor=tf.convert_to_tensor(test_pattern_images)
					testY_tensor=tf.convert_to_tensor(testY)

				print('Training example shape : '+str(train_animations.shape)+', '+str(train_pattern_images.shape))
				self.log.append('Training example shape : '+str(train_animations.shape)+', '+str(train_pattern_images.shape))
				print('Training label shape : '+str(trainY.shape))
				self.log.append('Training label shape : '+str(trainY.shape))
				print('Validation example shape : '+str(test_animations.shape)+', '+str(test_pattern_images.shape))
				self.log.append('Validation example shape : '+str(test_animations.shape)+', '+str(test_pattern_images.shape))
				print('Validation label shape : '+str(testY.shape))
				self.log.append('Validation label shape : '+str(testY.shape))
				print(datetime.datetime.now())
				self.log.append(str(datetime.datetime.now()))

				if dim_tconv<=16:
					batch_size=32
				elif dim_tconv<=64:
					batch_size=16
				elif dim_tconv<=128:
					batch_size=8
				else:
					batch_size=4

				model=self.combined_network(time_step=time_step,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,classes=len(self.classnames),level_tconv=level_tconv,level_conv=level_conv)
				self._compile_model(model,label_mode=label_mode,lambda_soft=lambda_soft)

				H=model.fit([train_animations,train_pattern_images],trainY,batch_size=batch_size,validation_data=([test_animations_tensor,test_pattern_images_tensor],testY_tensor),epochs=1000000,callbacks=self._standard_fit_callbacks(model_path,train_progress_cb,cancel_event))
				self._raise_if_fit_cancelled(cancel_event)

				self.save_categorizer_model(model,model_path)
				print('Trained Categorizer saved in: '+str(model_path))
				self.log.append('Trained Categorizer saved in: '+str(model_path))

				predictions=model.predict([test_animations,test_pattern_images],batch_size=batch_size)

				testY_hard=np.asarray(testY)
				C=len(self.classnames)
				if testY_hard.ndim==2 and testY_hard.shape[1]==2*C:
					testY_hard=testY_hard[:,:C]
				elif testY_hard.ndim==2 and C==2 and testY_hard.shape[1]==2:
					testY_hard=testY_hard[:,:1]

				if len(self.classnames)==2:
					predictions=[round(i[0]) for i in predictions]
					print(classification_report(testY_hard,predictions,target_names=self.classnames))
					report=classification_report(testY_hard,predictions,target_names=self.classnames,output_dict=True)
				else:
					print(classification_report(testY_hard.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames))
					report=classification_report(testY_hard.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames,output_dict=True)

				pd.DataFrame(report).transpose().to_csv(os.path.join(model_path,'training_metrics.csv'),float_format='%.2f')
				if out_path is not None:
					pd.DataFrame(report).transpose().to_excel(os.path.join(out_path,'training_metrics.xlsx'),float_format='%.2f')

				plt.style.use('classic')
				plt.figure()
				plt.plot(H.history['loss'],label='train_loss')
				plt.plot(H.history['val_loss'],label='val_loss')
				plt.plot(H.history['accuracy'],label='train_accuracy')
				plt.plot(H.history['val_accuracy'],label='val_accuracy')
				plt.title('Loss and Accuracy')
				plt.xlabel('Epoch')
				plt.ylabel('Loss/Accuracy')
				plt.legend(loc='center right')
				plt.savefig(os.path.join(model_path,'training_history.png'))
				if out_path is not None:
					plt.savefig(os.path.join(out_path,'training_history.png'))
					print('Training reports saved in: '+str(out_path))
					if len(self.log)>0:
						with open(os.path.join(out_path,'Training log.txt'),'w') as training_log:
							training_log.write('\n'.join(str(i) for i in self.log))
				plt.close('all')

			else:

				(train_files,test_files,_,_)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

				self.label_mode=label_mode
				self.lambda_soft=lambda_soft
				if str(label_mode)!='hard_only':
					class_means=self._class_mean_soft(data_path,list(self.classnames),soft_labels_path)
					if class_means is None:
						print('Soft labels not found; falling back to hard_only.')
						self.log.append('Soft labels not found; falling back to hard_only.')
						label_mode='hard_only'
						self.label_mode=label_mode
					else:
						print('Using soft-label mode: '+str(label_mode)+' (lambda_soft='+str(lambda_soft)+')')
						self.log.append('Using soft-label mode: '+str(label_mode))

				reuse=bool(skip_augment) and self.has_exported_aug_data(out_folder)
				if reuse:
					msg='Reusing existing augmented export (skip re-augment): '+str(out_folder)
					print(msg)
					self.log.append(msg)
					if progress_cb is not None:
						try:
							progress_cb(1,1,'Reusing existing augmented data…')
						except Exception:
							pass
				else:
					if skip_augment:
						msg='skip_augment requested but export incomplete; re-augmenting to: '+str(out_folder)
						print(msg)
						self.log.append(msg)
					print('Perform augmentation for the behavior examples and export them to: '+str(out_folder))
					self.log.append('Perform augmentation for the behavior examples and export them to: '+str(out_folder))
					print('This might take hours or days, depending on the capacity of your computer.')
					print(datetime.datetime.now())
					self.log.append(str(datetime.datetime.now()))

					print('Start to augment training examples...')
					self.log.append('Start to augment training examples...')
					train_folder=os.path.join(out_folder,'train')
					os.makedirs(train_folder,exist_ok=True)
					n_train=len(train_files)
					n_val=len(test_files)
					aug_total=n_train+n_val
					def _phase_cb(offset):
						if progress_cb is None:
							return None
						def _cb(done,tot,msg):
							progress_cb(offset+done,aug_total,msg)
						return _cb
					_,_,_=self.build_data(train_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=train_folder,num_workers=num_workers,progress_cb=_phase_cb(0),cancel_event=cancel_event)
					print('Start to augment validation examples...')
					self.log.append('Start to augment validation examples...')
					validation_folder=os.path.join(out_folder,'validation')
					os.makedirs(validation_folder,exist_ok=True)
					if augvalid:
						_,_,_=self.build_data(test_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=validation_folder,num_workers=num_workers,progress_cb=_phase_cb(n_train),cancel_event=cancel_event)
					else:
						_,_,_=self.build_data(test_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free,black_background=black_background,behavior_mode=behavior_mode,out_path=validation_folder,num_workers=num_workers,progress_cb=_phase_cb(n_train),cancel_event=cancel_event)

				self.train_combnet_onfly(
					out_folder,model_path,out_path=out_path,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,
					time_step=time_step,level_tconv=level_tconv,level_conv=level_conv,
					include_bodyparts=include_bodyparts,std=std,background_free=background_free,
					black_background=black_background,behavior_mode=behavior_mode,social_distance=social_distance,
					color_costar=color_costar,label_mode=label_mode,lambda_soft=lambda_soft,
					soft_labels_path=soft_labels_path,soft_source_path=data_path,
					train_progress_cb=train_progress_cb,cancel_event=cancel_event)


	def train_pattern_recognizer_onfly(self,data_path,model_path,out_path=None,dim=32,channel=3,time_step=15,level=2,include_bodyparts=True,std=0,background_free=True,black_background=True,behavior_mode=0,social_distance=0,label_mode='hard_only',lambda_soft=0.4,soft_labels_path=None,soft_source_path=None,train_progress_cb=None,cancel_event=None):

		# data_path: the folder that stores all the prepared training examples
		# model_path: the path to the trained Pattern Recognizer
		# out_path: if not None, will store the training reports in this folder
		# dim: the input dimension
		# channel: the input color channel, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation / pattern image, also the length of a behavior episode
		# level: complexity level, determines how deep the neural network is
		# include_bodyparts: whether to include body parts in the pattern images
		# std: a value between 0 and 255, higher value, less body parts will be included in the pattern images
		# background_free: whether to include background in animations
		# black_background: whether to set background black
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# social_distance: a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		# soft_source_path: original prepared-examples folder (for soft_labels.csv); defaults to data_path

		filters=8

		for i in range(round(dim/60)):
			filters=min(int(filters*2),64)

		inputs=Input(shape=(dim,dim,channel))

		print('Training Pattern Recognizer on-the-fly using the behavior examples in: '+str(data_path))
		self.log.append('Training Pattern Recognizer on-the-fly using the behavior examples in: '+str(data_path))
		print(datetime.datetime.now())
		self.log.append(str(datetime.datetime.now()))

		train_folder=os.path.join(data_path,'train')
		validation_folder=os.path.join(data_path,'validation')

		if os.path.isdir(train_folder) and os.path.isdir(validation_folder):

			if dim<=128:
				batch_size=32
			elif dim<=256:
				batch_size=16
			else:
				batch_size=8

			if behavior_mode==3:
				channel=channel
			else:
				channel=3

			# Probe class names first so soft class-means align
			_probe=DatasetFromPath(train_folder,batch_size=batch_size,dim_conv=dim,channel=channel)
			classnames=list(_probe.classmapping.keys())
			class_means=None
			effective_label_mode=label_mode
			if str(label_mode)!='hard_only':
				src=soft_source_path or data_path
				class_means=self._class_mean_soft(src,classnames,soft_labels_path)
				if class_means is None:
					print('Soft labels not found for onfly train; falling back to hard_only.')
					self.log.append('Soft labels not found for onfly train; falling back to hard_only.')
					effective_label_mode='hard_only'
				else:
					print('Using soft-label mode onfly: '+str(label_mode)+' (lambda_soft='+str(lambda_soft)+')')
					self.log.append('Using soft-label mode onfly: '+str(label_mode))

			train_data=DatasetFromPath(train_folder,batch_size=batch_size,dim_conv=dim,channel=channel,label_mode=effective_label_mode,class_means=class_means)
			validation_data=DatasetFromPath(validation_folder,batch_size=batch_size,dim_conv=dim,channel=channel,label_mode=effective_label_mode,class_means=class_means)


			if include_bodyparts:
				inner_code=0
			else:
				inner_code=1

			if background_free:
				background_code=0
			else:
				background_code=1

			if black_background:
				black_code=0
			else:
				black_code=1

			if behavior_mode>=3:
				time_step=std=0
				inner_code=1

			parameters={'classnames':list(train_data.classmapping.keys()),'dim_conv':int(dim),'channel':int(channel),'time_step':int(time_step),'network':0,'level_conv':int(level),'inner_code':int(inner_code),'std':int(std),'background_free':int(background_code),'black_background':int(black_code),'behavior_kind':int(behavior_mode),'social_distance':int(social_distance),'label_mode':str(effective_label_mode),'lambda_soft':float(lambda_soft)}
			pd_parameters=pd.DataFrame.from_dict(parameters)
			pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

			self.classnames=list(train_data.classmapping.keys())
			if level<5:
				model=self.simple_vgg(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
			else:
				model=self.simple_resnet(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
			self._compile_model(model,label_mode=effective_label_mode,lambda_soft=lambda_soft)

			H=model.fit(train_data,validation_data=(validation_data),epochs=1000000,callbacks=self._standard_fit_callbacks(model_path,train_progress_cb,cancel_event))
			self._raise_if_fit_cancelled(cancel_event)

			self.save_categorizer_model(model,model_path)
			print('Trained Categorizer saved in: '+str(model_path))
			self.log.append('Trained Categorizer saved in: '+str(model_path))
			print(datetime.datetime.now())
			self.log.append(str(datetime.datetime.now()))

			plt.style.use('classic')
			plt.figure()
			plt.plot(H.history['loss'],label='train_loss')
			plt.plot(H.history['val_loss'],label='val_loss')
			plt.plot(H.history['accuracy'],label='train_accuracy')
			plt.plot(H.history['val_accuracy'],label='val_accuracy')
			plt.title('Loss and Accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Loss/Accuracy')
			plt.legend(loc='center right')
			plt.savefig(os.path.join(model_path,'training_history.png'))
			if out_path is not None:
				plt.savefig(os.path.join(out_path,'training_history.png'))
				print('Training reports saved in: '+str(out_path))
				if len(self.log)>0:
					with open(os.path.join(out_path,'Training log.txt'),'w') as training_log:
						training_log.write('\n'.join(str(i) for i in self.log))
			plt.close('all')

		else:

			print('No train / validation folder!')


	def train_animation_analyzer_onfly(self,data_path,model_path,out_path=None,dim=32,channel=1,time_step=15,level=2,include_bodyparts=True,std=0,background_free=True,black_background=True,behavior_mode=0,social_distance=0,color_costar=False,train_progress_cb=None,cancel_event=None):

		# data_path: the folder that stores all the prepared training examples
		# model_path: the path to the trained Animation Analyzer
		# out_path: if not None, will store the training reports in this folder
		# dim: the input dimension of Animation Analyzer
		# channel: the input color channel of Animation Analyzer, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation, also the input length of Animation Analyzer
		# level: complexity level of Animation Analyzer, determines how deep the neural network is
		# include_bodyparts: whether to include body parts in the pattern images
		# std: a value between 0 and 255, higher value, less body parts will be included in the pattern images
		# background_free: whether to include background in animations
		# black_background: whether to set background
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# social_distance: a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		# color_costar: in 'interactive advanced' mode, whether to make the supporting roles RGB scale in animations

		filters=8

		for i in range(round(dim/60)):
			filters=min(int(filters*2),64)

		inputs=Input(shape=(dim,dim,channel))

		print('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))
		self.log.append('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))
		print(datetime.datetime.now())
		self.log.append(str(datetime.datetime.now()))

		train_folder=os.path.join(data_path,'train')
		validation_folder=os.path.join(data_path,'validation')

		if os.path.isdir(train_folder) and os.path.isdir(validation_folder):

			if dim<=16:
				batch_size=32
			elif dim<=64:
				batch_size=16
			elif dim<=128:
				batch_size=8
			else:
				batch_size=4

			train_data=DatasetFromPath_AA(train_folder,length=time_step,batch_size=batch_size,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel)
			validation_data=DatasetFromPath_AA(validation_folder,length=time_step,batch_size=batch_size,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel)

			if include_bodyparts:
				inner_code=0
			else:
				inner_code=1

			if background_free:
				background_code=0
			else:
				background_code=1

			if black_background:
				black_code=0
			else:
				black_code=1

			if color_costar:
				color_code=0
			else:
				color_code=1

			parameters={'classnames':list(train_data.classmapping.keys()),'dim_tconv':int(dim),'channel':int(channel),'time_step':int(time_step),'network':1,'level_tconv':int(level),'inner_code':int(inner_code),'std':int(std),'background_free':int(background_code),'black_background':int(black_code),'behavior_kind':int(behavior_mode),'social_distance':int(social_distance),'color_code':int(color_code)}
			pd_parameters=pd.DataFrame.from_dict(parameters)
			pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)


			if level<5:
				model=self.simple_tvgg(inputs,filters,classes=len(list(train_data.classmapping.keys())),level=level,with_classifier=True)
			else:
				model=self.simple_tresnet(inputs,filters,classes=len(list(train_data.classmapping.keys())),level=level,with_classifier=True)
			if len(list(train_data.classmapping.keys()))==2:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='binary_crossentropy',metrics=['accuracy'])
			else:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

			H=model.fit(train_data,validation_data=(validation_data),epochs=1000000,callbacks=self._standard_fit_callbacks(model_path,train_progress_cb,cancel_event))
			self._raise_if_fit_cancelled(cancel_event)

			self.save_categorizer_model(model,model_path)
			print('Trained Categorizer saved in: '+str(model_path))
			self.log.append('Trained Categorizer saved in: '+str(model_path))
			print(datetime.datetime.now())
			self.log.append(str(datetime.datetime.now()))

			plt.style.use('classic')
			plt.figure()
			plt.plot(H.history['loss'],label='train_loss')
			plt.plot(H.history['val_loss'],label='val_loss')
			plt.plot(H.history['accuracy'],label='train_accuracy')
			plt.plot(H.history['val_accuracy'],label='val_accuracy')
			plt.title('Loss and Accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Loss/Accuracy')
			plt.legend(loc='center right')
			plt.savefig(os.path.join(model_path,'training_history.png'))
			if out_path is not None:
				plt.savefig(os.path.join(out_path,'training_history.png'))
				print('Training reports saved in: '+str(out_path))
				if len(self.log)>0:
					with open(os.path.join(out_path,'Training log.txt'),'w') as training_log:
						training_log.write('\n'.join(str(i) for i in self.log))
			plt.close('all')

		else:

			print('No train / validation folder!')


	def train_combnet_onfly(self,data_path,model_path,out_path=None,dim_tconv=32,dim_conv=64,channel=1,time_step=15,level_tconv=1,level_conv=2,include_bodyparts=True,std=0,background_free=True,black_background=True,behavior_mode=0,social_distance=0,color_costar=False,label_mode='hard_only',lambda_soft=0.4,soft_labels_path=None,soft_source_path=None,train_progress_cb=None,cancel_event=None):

		# data_path: the folder that stores all the prepared training examples
		# model_path: the path to the trained Animation Analyzer
		# out_path: if not None, will store the training reports in this folder
		# dim_tconv: the input dimension of Animation Analyzer
		# dim_conv: the input dimension of Pattern Recognizer
		# channel: the input color channel of Animation Analyzer, 1 is gray scale, 3 is RGB
		# time_step: the duration of an animation, also the input length of Animation Analyzer
		# level_tconv: complexity level of Animation Analyzer, determines how deep the neural network is
		# level_conv: complexity level of Pattern Recognizer, determines how deep the neural network is
		# include_bodyparts: whether to include body parts in the pattern images
		# std: a value between 0 and 255, higher value, less body parts will be included in the pattern images
		# background_free: whether to include background in animations
		# black_background: whether to set background
		# behavior_mode:  0--non-interactive, 1--interactive basic, 2--interactive advanced, 3--static images
		# social_distance: a threshold (folds of size of a single animal) on whether to include individuals that are not main character in behavior examples
		# color_costar: in 'interactive advanced' mode, whether to make the supporting roles RGB scale in animations
		# soft_source_path: original prepared-examples folder (for soft_labels.csv); defaults to data_path

		print('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))
		self.log.append('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))
		print(datetime.datetime.now())
		self.log.append(str(datetime.datetime.now()))

		train_folder=os.path.join(data_path,'train')
		validation_folder=os.path.join(data_path,'validation')

		if os.path.isdir(train_folder) and os.path.isdir(validation_folder):

			if dim_tconv<=16:
				batch_size=32
			elif dim_tconv<=64:
				batch_size=16
			elif dim_tconv<=128:
				batch_size=8
			else:
				batch_size=4

			_probe=DatasetFromPath_AA(train_folder,length=time_step,batch_size=batch_size,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel)
			classnames=list(_probe.classmapping.keys())
			class_means=None
			effective_label_mode=label_mode
			if str(label_mode)!='hard_only':
				src=soft_source_path or data_path
				class_means=self._class_mean_soft(src,classnames,soft_labels_path)
				if class_means is None:
					print('Soft labels not found for onfly train; falling back to hard_only.')
					self.log.append('Soft labels not found for onfly train; falling back to hard_only.')
					effective_label_mode='hard_only'
				else:
					print('Using soft-label mode onfly: '+str(label_mode)+' (lambda_soft='+str(lambda_soft)+')')
					self.log.append('Using soft-label mode onfly: '+str(label_mode))

			train_data=DatasetFromPath_AA(train_folder,length=time_step,batch_size=batch_size,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,label_mode=effective_label_mode,class_means=class_means)
			validation_data=DatasetFromPath_AA(validation_folder,length=time_step,batch_size=batch_size,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,label_mode=effective_label_mode,class_means=class_means)

			if include_bodyparts:
				inner_code=0
			else:
				inner_code=1

			if background_free:
				background_code=0
			else:
				background_code=1

			if black_background:
				black_code=0
			else:
				black_code=1

			if color_costar:
				color_code=0
			else:
				color_code=1

			self.classnames=list(train_data.classmapping.keys())
			parameters={'classnames':list(self.classnames),'dim_tconv':int(dim_tconv),'dim_conv':int(dim_conv),'channel':int(channel),'time_step':int(time_step),'network':2,'level_tconv':int(level_tconv),'level_conv':int(level_conv),'inner_code':int(inner_code),'std':int(std),'background_free':int(background_code),'black_background':int(black_code),'behavior_kind':int(behavior_mode),'social_distance':int(social_distance),'color_code':int(color_code),'label_mode':str(effective_label_mode),'lambda_soft':float(lambda_soft)}
			pd_parameters=pd.DataFrame.from_dict(parameters)
			pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

			model=self.combined_network(time_step=time_step,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,classes=len(self.classnames),level_tconv=level_tconv,level_conv=level_conv)
			self._compile_model(model,label_mode=effective_label_mode,lambda_soft=lambda_soft)

			H=model.fit(train_data,validation_data=(validation_data),epochs=1000000,callbacks=self._standard_fit_callbacks(model_path,train_progress_cb,cancel_event))
			self._raise_if_fit_cancelled(cancel_event)

			self.save_categorizer_model(model,model_path)
			print('Trained Categorizer saved in: '+str(model_path))
			self.log.append('Trained Categorizer saved in: '+str(model_path))
			print(datetime.datetime.now())
			self.log.append(str(datetime.datetime.now()))

			plt.style.use('classic')
			plt.figure()
			plt.plot(H.history['loss'],label='train_loss')
			plt.plot(H.history['val_loss'],label='val_loss')
			plt.plot(H.history['accuracy'],label='train_accuracy')
			plt.plot(H.history['val_accuracy'],label='val_accuracy')
			plt.title('Loss and Accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Loss/Accuracy')
			plt.legend(loc='center right')
			plt.savefig(os.path.join(model_path,'training_history.png'))
			if out_path is not None:
				plt.savefig(os.path.join(out_path,'training_history.png'))
				print('Training reports saved in: '+str(out_path))
				if len(self.log)>0:
					with open(os.path.join(out_path,'Training log.txt'),'w') as training_log:
						training_log.write('\n'.join(str(i) for i in self.log))
			plt.close('all')

		else:

			print('No train / validation folder!')


	def test_categorizer(self,groundtruth_path,model_path,result_path=None):

		# groundtruth_path: the folder that stores all the groundtruth behavior examples, each subfolder should be a behavior category, all categories must match those in the Categorizer
		# model_path: path to the Categorizer
		# result_path: if not None, will store the testing reports in this folder

		print('Testing the selected Categorizer...')

		animations=deque()
		pattern_images=deque()
		labels=deque()

		parameters=pd.read_csv(os.path.join(model_path,'model_parameters.txt'))

		if 'dim_conv' in list(parameters.keys()):
			dim_conv=int(parameters['dim_conv'][0])
		if 'dim_tconv' in list(parameters.keys()):
			dim_tconv=int(parameters['dim_tconv'][0])
		if 'level_conv' in list(parameters.keys()):
			level_conv=int(parameters['level_conv'][0])
		if 'dim_tconv' in list(parameters.keys()):
			level_tconv=int(parameters['level_tconv'][0])
		if 'channel' in list(parameters.keys()):
			channel=int(parameters['channel'][0])
		if 'behavior_kind' in list(parameters.keys()):
			behavior_mode=int(parameters['behavior_kind'][0])
		else:
			behavior_mode=0
		if behavior_mode==0:
			print('The behavior mode of the Categorizer: Non-interactive.')
		elif behavior_mode==1:
			print('The behavior mode of the Categorizer: Interactive basic.')
		elif behavior_mode==2:
			print('The behavior mode of the Categorizer: Interactive advanced (Social distance '+str(parameters['social_distance'][0])+').')
			if 'color_code' in list(parameters.keys()):
				color_code=int(parameters['color_code'][0])
				if color_code==0:
					print('The Categorizer recognizes RGB scale main character and RGB scale  supporting characters.')
				else:
					print('The Categorizer recognizes RGB scale main character and grayscale supporting characters.')
			else:
				print('The Categorizer recognizes RGB scale main character and grayscale supporting characters.')
		else:
			print('The behavior mode of the Categorizer: Static images (non-interactive).')
		network=int(parameters['network'][0])
		if network==0:
			if behavior_mode==3:
				print('The type of the Categorizer: Pattern Recognizer (Lv '+str(level_conv)+'; Shape '+str(dim_conv)+' X '+str(dim_conv)+' X '+str(channel)+').')
			else:
				print('The type of the Categorizer: Pattern Recognizer (Lv '+str(level_conv)+'; Shape '+str(dim_conv)+' X '+str(dim_conv)+' X 3).')
		if network==1:
			print('The type of the Categorizer: Animation Analyzer (Lv '+str(level_tconv)+'; Shape '+str(dim_tconv)+' X '+str(channel)+').')
		if network==2:
			print('The type of the Categorizer: Animation Analyzer (Lv '+str(level_tconv)+'; Shape '+str(dim_tconv)+' X '+str(dim_tconv)+' X '+str(channel)+') + Pattern Recognizer (Lv '+str(level_conv)+'; Shape '+str(dim_conv)+' X '+str(dim_conv)+' X 3).')
		length=int(parameters['time_step'][0])
		print('The length of a behavior example in the Categorizer: '+str(length)+' frames.')
		if int(parameters['inner_code'][0])==0:
			print('The Categorizer includes body parts in analysis with STD = '+str(parameters['std'][0])+'.')
		else:
			print('The Categorizer does not include body parts in analysis.')
		if int(parameters['background_free'][0])==0:
			print('The Categorizer does not include background in analysis.')
		else:
			print('The Categorizer includes background in analysis.')
		if 'black_background' in parameters:
			if int(parameters['black_background'][0])==1:
				print('The background is white in the Categorizer.')
		classnames=list(parameters['classnames'])
		classnames=[str(i) for i in classnames]
		print('Behavior names in the Categorizer: '+str(classnames))
		behaviornames=[i for i in os.listdir(groundtruth_path) if os.path.isdir(os.path.join(groundtruth_path,i))]
		incorrect_behaviors=list(set(behaviornames)-set(classnames))
		incorrect_classes=list(set(classnames)-set(behaviornames))
		if len(incorrect_behaviors)>0:
			print('Mismatched behavior names in testing examples: '+str(incorrect_behaviors))
		if len(incorrect_classes)>0:
			print('Unused behavior names in the Categorizer: '+str(incorrect_classes))

		if len(incorrect_behaviors)==0 and len(incorrect_classes)==0:

			for behavior in behaviornames:

				if network!=0:
					filenames=[i for i in os.listdir(os.path.join(groundtruth_path,behavior)) if i.endswith('.avi')]
				else:
					filenames=[i for i in os.listdir(os.path.join(groundtruth_path,behavior)) if i.endswith('.jpg')]

				for i in filenames:

					if network!=0:

						path_to_animation=os.path.join(groundtruth_path,behavior,i)

						capture=cv2.VideoCapture(path_to_animation)
						animation=deque()
						frames=deque(maxlen=length)

						while True:
							retval,frame=capture.read()
							if frame is None:
								break
							frames.append(frame)

						capture.release()

						for frame in frames:
							frame=np.uint8(exposure.rescale_intensity(frame,out_range=(0,255)))
							if channel==1:
								frame=cv2.cvtColor(np.uint8(frame),cv2.COLOR_BGR2GRAY)
							frame=cv2.resize(frame,(dim_tconv,dim_tconv),interpolation=cv2.INTER_AREA)
							frame=img_to_array(frame)
							animation.append(frame)

						animations.append(np.array(animation))

					if network!=1:

						path_to_pattern_image=os.path.splitext(os.path.join(groundtruth_path,behavior,i))[0]+'.jpg'
						pattern_image=cv2.imread(path_to_pattern_image)
						if behavior_mode==3:
							if channel==1:
								pattern_image=cv2.cvtColor(pattern_image,cv2.COLOR_BGR2GRAY)
						pattern_image=cv2.resize(pattern_image,(dim_conv,dim_conv),interpolation=cv2.INTER_AREA)
						pattern_images.append(img_to_array(pattern_image))

					labels.append(classnames.index(behavior))

			if network!=0:
				animations=np.array(animations,dtype='float32')/255.0
			pattern_images=np.array(pattern_images,dtype='float32')/255.0

			labels=np.array(labels)

			model=self.load_categorizer_model(model_path)

			if network==0:
				predictions=model.predict(pattern_images,batch_size=32)
			elif network==1:
				predictions=model.predict(animations,batch_size=32)
			else:
				predictions=model.predict([animations,pattern_images],batch_size=32)

			if len(classnames)==2:
				predictions=[round(i[0]) for i in predictions]
				print(classification_report(labels,predictions,target_names=classnames))
				report=classification_report(labels,predictions,target_names=classnames,output_dict=True)
			else:
				print(classification_report(labels,predictions.argmax(axis=1),target_names=classnames))
				report=classification_report(labels,predictions.argmax(axis=1),target_names=classnames,output_dict=True)

			if result_path is not None:
				pd.DataFrame(report).transpose().to_excel(os.path.join(result_path,'testing_reports.xlsx'),float_format='%.2f')

			print('Testing completed!')
