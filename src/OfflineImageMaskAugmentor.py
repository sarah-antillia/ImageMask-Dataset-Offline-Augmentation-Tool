# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# OfflineImageMaskAugmentor.py
# 2024/05/10 Offline Augmentation Tool for ImageMask Dataset for Image-Segmentation

"""
"""

"""
Input dataset_dir takes the following structure
./dataset_dir
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks

"""

"""
Augemented dataset_dir will take the following folder structure.

./augmented_dir
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks

"""

import os
import sys
import glob
import shutil

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from ConfigParser import ConfigParser
import traceback

class OfflineImageMaskAugmentor:
  
  def __init__(self, config_file):
    self.seed     = 137
    self.config   = ConfigParser(config_file)
    self.config.dump_all()

    #self.debug    = self.config.get(ConfigParser.GENERATOR, "debug",  dvalue=True)
    self.W        = self.config.get(ConfigParser.DATASET,     "image_width")
    self.H        = self.config.get(ConfigParser.DATASET,     "image_height")
    #Specify dataset name for example 'test' to exclude from augmentation processing.    
    self.no_augmentation = self.config.get(ConfigParser.DATASET,  "no_augmentation", dvalue="test" )
    self.dataset_dir = self.config.get(ConfigParser.DATASET,  "dataset_dir") 
    self.augmented_dir = self.config.get(ConfigParser.DATASET, "augmented_dir")

    self.rotation = self.config.get(ConfigParser.AUGMENTOR, "rotation", dvalue=True)
    self.SHRINKS  = self.config.get(ConfigParser.AUGMENTOR, "shrinks",  dvalue=[0.8])
    self.ANGLES   = self.config.get(ConfigParser.AUGMENTOR, "angles",   dvalue=[90, 180, 270])

    self.SHEARS   = self.config.get(ConfigParser.AUGMENTOR, "shears",   dvalue=[])

    self.hflip    = self.config.get(ConfigParser.AUGMENTOR, "hflip", dvalue=True)
    self.vflip    = self.config.get(ConfigParser.AUGMENTOR, "vflip", dvalue=True)
  
    self.deformation = self.config.get(ConfigParser.AUGMENTOR, "deformation", dvalue=False)
    if self.deformation:
      self.alpha    = self.config.get(ConfigParser.DEFORMATION, "alpah", dvalue=1300)
      self.sigmoid  = self.config.get(ConfigParser.DEFORMATION, "sigmoid", dvalue=8)
 
    self.distortion = self.config.get(ConfigParser.AUGMENTOR, "distortion", dvalue=False)
    # Distortion
    if self.distortion:
      self.gaussina_filer_rsigma = self.config.get(ConfigParser.DISTORTION, "gaussian_filter_rsigma", dvalue=40)
      self.gaussina_filer_sigma  = self.config.get(ConfigParser.DISTORTION, "gaussian_filter_sigma",  dvalue=0.5)
      self.distortions           = self.config.get(ConfigParser.DISTORTION, "distortions",  dvalue=[0.02])
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)


    if not os.path.exists(self.dataset_dir):
      error = "NOT FOUND " + self.dataset_dir
      raise Exception(error)
    
    if os.path.exists(self.augmented_dir):
      shutil.rmtree(self.augmented_dir)

    if not os.path.exists(self.augmented_dir):
      os.makedirs(self.augmented_dir)

  def augment(self):
    sub_datasets = os.listdir(self.dataset_dir)
    for sub_dataset in sub_datasets:
      # sub_dataset takes a value of 'test', 'train' and 'valid'
      images_dir  = os.path.join(self.dataset_dir, sub_dataset + "/images")
      masks_dir   = os.path.join(self.dataset_dir, sub_dataset + "/masks")
      self.augment_one(sub_dataset, images_dir, masks_dir)

  def augment_one(self, sub_dataset, images_dir, masks_dir):
    output_images_dir = os.path.join(self.augmented_dir, sub_dataset + "/images")
    output_masks_dir = os.path.join(self.augmented_dir, sub_dataset + "/masks")
  
    image_files  = glob.glob(images_dir + "/*.jpg")
    image_files += glob.glob(images_dir + "/*.png")
    image_files += glob.glob(images_dir + "/*.tif")
    image_files += glob.glob(images_dir + "/*.bmp")
      
    mask_files  = glob.glob(masks_dir + "/*.jpg")
    mask_files += glob.glob(masks_dir + "/*.png")
    mask_files += glob.glob(masks_dir + "/*.tif")
    mask_files += glob.glob(masks_dir + "/*.bmp")

    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    num_image_files = len(image_files)
    for i in range(num_image_files):
      image = cv2.imread(image_files[i])
      mask  = cv2.imread(mask_files[i])

      image = cv2.resize(image, (self.W, self.H))
      mask  = cv2.resize(mask,  (self.W, self.H))

      image_basename = os.path.basename(image_files[i])
      mask_basename  = os.path.basename(mask_files[i]) 

      filepath = os.path.join(output_images_dir, image_basename)
      cv2.imwrite(filepath, image)

      filepath = os.path.join(output_masks_dir,  mask_basename)
      cv2.imwrite(filepath, mask)

      if sub_dataset != self.no_augmentation:
        self.augment_image_and_mask(image, mask,
                output_images_dir, image_basename,
                output_masks_dir,  mask_basename )

  # It applies  horizotanl and vertical flipping operations to image and mask repectively.
  def augment_image_and_mask(self, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    if self.hflip:
      hflip_image = self.horizontal_flip(image) 
      hflip_mask  = self.horizontal_flip(mask) 
      #print("--- hflp_mask shape {}".format(hflip_mask.shape))
      filepath = os.path.join(generated_images_dir, "hfliped_" + image_basename)
      cv2.imwrite(filepath, hflip_image)
      print("=== Saved {}".format(filepath))
      filepath = os.path.join(generated_masks_dir,  "hfliped_" + mask_basename)
      cv2.imwrite(filepath, hflip_mask)
      print("=== Saved {}".format(filepath))

    if self.vflip:
      vflip_image = self.vertical_flip(image)
      vflip_mask  = self.vertical_flip(mask)
      
      filepath = os.path.join(generated_images_dir, "vfliped_" + image_basename)
      cv2.imwrite(filepath, vflip_image)
      print("=== Saved {}".format(filepath))

      filepath = os.path.join(generated_masks_dir,  "vfliped_" + mask_basename)
      cv2.imwrite(filepath, vflip_mask)
      print("=== Saved {}".format(filepath))

    if self.rotation:
       self.rotate(image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
       
    if type(self.SHRINKS) is list and len(self.SHRINKS)>0:
       self.shrink(image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if type(self.SHEARS) is list and len(self.SHEARS)>0:
       self.shear(image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if self.deformation:
      self.deform(image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
      
    if self.distortion:
      self.distort(image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )


  def horizontal_flip(self, image): 
    image = image[:, ::-1, :]
    return image

  def vertical_flip(self, image):
    image = image[::-1, :, :]
    return image
  
  def rotate(self, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    for angle in self.ANGLES:      

      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
      color  = image[2][2].tolist()
      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=color)
      color  = mask[2][2].tolist()
      rotated_mask  = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=(self.W, self.H), borderValue=color)
       
      #rotated_mask  = np.expand_dims(rotated_mask, axis=-1) 

      filepath = os.path.join(generated_images_dir, "rotated_" + str(angle) + "_" + image_basename)
      cv2.imwrite(filepath, rotated_image)
      print("=== Saved {}".format(filepath))
      filepath = os.path.join(generated_masks_dir,  "rotated_" + str(angle) + "_" + mask_basename)
      cv2.imwrite(filepath, rotated_mask)
      print("=== Saved {}".format(filepath))
  

  def shrink(self, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):

    h, w = image.shape[:2]
  
    for shrink in self.SHRINKS:
      rw = int (w * shrink)
      rh = int (h * shrink)
      resized_image = cv2.resize(image, dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      resized_mask  = cv2.resize(mask,  dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      
      squared_image = self.paste(resized_image, mask=False)
      squared_mask  = self.paste(resized_mask,  mask=True)

      ratio   = str(shrink).replace(".", "_")
      image_filename = "shrinked_" + ratio + "_" + image_basename
      filepath  = os.path.join(generated_images_dir, image_filename)
      cv2.imwrite(filepath, squared_image)
      #print("=== Saved {}".format(image_filepath))
      print("=== Saved {}".format(filepath))
    
      mask_filename = "shrinked_" + ratio + "_" + mask_basename
      filepath  = os.path.join(generated_masks_dir, mask_filename)
      cv2.imwrite(filepath, squared_mask)
      print("=== Saved {}".format(filepath))


  def paste(self, image, mask=False):
    l = len(image.shape)
   
    h, w,  = image.shape[:2]
    if l==3:
      back_color = image[2][2]
      background = np.ones((self.H, self.W, 3), dtype=np.uint8)
      background = background * back_color

      #background = np.zeros((self.H, self.W, 3), dtype=np.uint8)

      (b, g, r) = image[h-10][w-10] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      background += [b, g, r][::-1]
    else:
      v =  image[h-10][w-10] 
      image  = np.expand_dims(image, axis=-1) 
      background = np.zeros((self.H, self.W, 1), dtype=np.uint8)
      background[background !=v] = v
    x = (self.W - w)//2
    y = (self.H - h)//2
    background[y:y+h, x:x+w] = image
    return background
  

  # This method has been taken from the following code in stackoverflow.
  # https://stackoverflow.com/questions/57881430/how-could-i-implement-a-centered-shear-an-image-with-opencv
  # Do shear, hflip and vflip the image and mask and save them 
  def shear(self, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename ):

    if self.SHEARS == None or len(self.SHEARS) == 0:
      return
   
    H, W = image.shape[:2]
    for shear in self.SHEARS:
      ratio = str(shear).replace(".", "_")
      M2 = np.float32([[1, 0, 0], [shear, 1,0]])
      M2[0,2] = -M2[0,1] * H/2 
      M2[1,2] = -M2[1,0] * W/2 

      # 1 shear and image
      color  = image[2][2].tolist()
      sheared_image = cv2.warpAffine(image, M2, (W, H), borderValue=color)

      color  = mask[2][2].tolist()
      sheared_mask  = cv2.warpAffine(mask,  M2, (W, H), borderValue=color)
      #sheared_mask  = np.expand_dims(sheared_mask, axis=-1) 

      # 2 save sheared image and mask 
      filepath = os.path.join(generated_images_dir, "sheared_" + ratio + "_" + image_basename)
      cv2.imwrite(filepath, sheared_image)
      print("=== Saved {}".format(filepath))

      filepath = os.path.join(generated_masks_dir,  "sheared_" + ratio + "_" + mask_basename)
      cv2.imwrite(filepath, sheared_mask)
      print("=== Saved {}".format(filepath))
  
      if self.hflip:
        # hflipp sheared image and mask
        hflipped_sheared_image  = self.horizontal_flip(sheared_image)
        hflipped_sheared_mask   = self.horizontal_flip(sheared_mask)
        filepath = os.path.join(generated_images_dir, "hflipped_sheared_" + ratio + "_" + image_basename)
        cv2.imwrite(filepath, hflipped_sheared_image)
        print("=== Saved {}".format(filepath))

        filepath = os.path.join(generated_masks_dir,  "hflipped_sheared_" + ratio + "_" + mask_basename)
        cv2.imwrite(filepath, hflipped_sheared_mask)
        print("=== Saved {}".format(filepath))

      if self.vflip:

        vflipped_sheared_image  = self.vertical_flip(sheared_image)
        vflipped_sheared_mask   = self.vertical_flip(sheared_mask)

        filepath = os.path.join(generated_images_dir, "vflipped_sheared_" + ratio + "_" + image_basename)
        cv2.imwrite(filepath, vflipped_sheared_image)
        print("=== Saved {}".format(filepath))
        filepath = os.path.join(generated_masks_dir,  "vflipped_sheared_" + ratio + "_" + mask_basename)
        cv2.imwrite(filepath, vflipped_sheared_mask)
        print("=== Saved {}".format(filepath))

        
      if self.hflip and self.vflip:
        hvflipped_sheared_image = self.vertical_flip(hflipped_sheared_image)
        hvflipped_sheared_mask  = self.vertical_flip(hflipped_sheared_mask)


        filepath = os.path.join(generated_images_dir, "hvflipped_sheared_" + ratio + "_" + image_basename)
        cv2.imwrite(filepath, hvflipped_sheared_image)
        print("=== Saved {}".format(filepath))
        filepath = os.path.join(generated_masks_dir,  "hvflipped_sheared_" + ratio + "_" + mask_basename)
        cv2.imwrite(filepath, hvflipped_sheared_mask)
        print("=== Saved {}".format(filepath))


  # This method has been taken from the following code.
  # https://github.com/MareArts/Elastic_Effect/blob/master/Elastic.py
  #
  # https://cognitivemedium.com/assets/rmnist/Simard.pdf
  #
  # See also
  # https://www.kaggle.com/code/jiqiujia/elastic-transform-for-data-augmentation/notebook
  # 
  def deform(self, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
    deformed_image = deformed_image.reshape(image.shape)

    shape = mask.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    deformed_mask = map_coordinates(mask, indices, order=1, mode='nearest')  
    deformed_mask = deformed_mask.reshape(mask.shape)

    image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(self.sigmoid) + "_" + image_basename
    image_filepath  = os.path.join(generated_images_dir, image_filename)
    cv2.imwrite(image_filepath, deformed_image)
    print("=== Saved {}".format(image_filepath))
    
    mask_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(self.sigmoid) + "_" + mask_basename
    mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
    cv2.imwrite(mask_filepath, deformed_mask)
    print("=== Saved {}".format(mask_filepath))

  # The code used here is based on the following stakoverflow web-site
  #https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/78031420#78031420

  def distort(self, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename):
    for size in self.distortions:
      distorted_image = self.distort_one(image, size)  
      distorted_image = distorted_image.reshape(image.shape)
      distorted_mask  = self.distort_one(mask, size)
      distorted_mask  = distorted_mask.reshape(mask.shape)

      image_filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + image_basename

      image_filepath  = os.path.join(generated_images_dir, image_filename)
      cv2.imwrite(image_filepath, distorted_image)
      print("=== Saved {}".format(image_filepath))
    
      mask_filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + mask_basename
      mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
      cv2.imwrite(mask_filepath, distorted_mask)
      print("=== Saved {}".format(mask_filepath))

  def distort_one(self, image, size):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)

    dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
    dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
    sizex = int(xsize * size)
    sizey = int(xsize * size)
    dx *= sizex/dx.max()  
    dy *= sizey/dy.max()

    img = gaussian_filter(image, self.gaussina_filer_sigma)

    yy, xx = np.indices(shape)
    xmap = (xx-dx).astype(np.float32)
    ymap = (yy-dy).astype(np.float32)

    distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
    distorted = cv2.resize(distorted, (w, h))
    return distorted


if __name__ == "__main__":
  try:
    config_file = "generator.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      error = "NOT FOUND: " + config_file
      raise Exception(error)
    augmentor = OfflineImageMaskAugmentor(config_file)
    augmentor.augment()

  except:
     traceback.print_exc()
