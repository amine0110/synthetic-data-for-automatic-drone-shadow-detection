
"""
Created by Mohammed El Amine Mokhtari
ISIA Lab, Mons University, Belgium


© - 2022 – UMONS-Numediart
Synthetic data generation –  is free software : you can redistribute it and/or modify it under the terms of the PSF License. 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the PSF License for more details.  You should have received a copy of the … License along with this program.  
If not, see Link to the license.
https://docs.python.org/3/license.html#psf-license-agreement-for-python-release

"""


from PIL import Image, ImageFilter, ImageEnhance
from glob import glob
import random
from tqdm import tqdm
import xml.dom.minidom as md
import os
import cv2


def remove_files(in_dir, xml=False, jpg=False, png=False, txt=False):
    if xml: data = glob(os.path.join(in_dir, '*.xml'))
    if jpg: data = glob(os.path.join(in_dir, '*.jpg'))
    if png: data = glob(os.path.join(in_dir, '*.png'))
    if txt: data = glob(os.path.join(in_dir, '*.txt'))

    for file in data:
        os.remove(file)
    print('Files deleted successfully')


def image_over_image(f_img, b_img, save_path, resize_value=0, show=False, opacity=1, position = (0,0), blur=False, save=False):
    front_image = Image.open(f_img) # The shadow image
    back_image= Image.open(b_img) # The scene image
    
    # Resize the shadow image if needed then convert back/front images into RGBA
    front_image = front_image.resize((front_image.width+resize_value, front_image.height+resize_value), Image.ANTIALIAS)
    front_image = front_image.convert('RGBA')
    back_image = back_image.convert('RGBA')
    
    # Alpha controls the value of the opacity
    alpha = front_image.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    front_image.putalpha(alpha)
    
    # Past the shadow on the scene
    back_image.paste(front_image, position, front_image)
    back_image = back_image.convert('RGB')
    
    if blur: back_image = back_image.filter(ImageFilter.BLUR) # Add blur if needed
        
    if show: back_image.show() # To show the image if you don't want to save it
    
    if save: back_image.save(save_path) # To save the image
        
    return back_image # Return it anyway


def fill_xml(path_save, width, height, depth, xmin, ymin, xmax, ymax, path_image, filename, object_name):
    file = md.parse( "C:\\Users\\amine\\Documents\\Amine_Files\\PhD\\codes\\images\\trans2\\image0.xml" )  # Path to a prexisting xml file

    file.getElementsByTagName( "width" )[0].firstChild.nodeValue = str(width)
    file.getElementsByTagName( "height" )[0].firstChild.nodeValue = str(height)
    file.getElementsByTagName( "depth" )[0].firstChild.nodeValue = str(depth)
    file.getElementsByTagName( "xmin" )[0].firstChild.nodeValue = str(xmin)
    file.getElementsByTagName( "ymin" )[0].firstChild.nodeValue = str(ymin)
    file.getElementsByTagName( "xmax" )[0].firstChild.nodeValue = str(xmax)
    file.getElementsByTagName( "ymax" )[0].firstChild.nodeValue = str(ymax)
    file.getElementsByTagName( "path" )[0].firstChild.nodeValue = path_image
    file.getElementsByTagName( "filename" )[0].firstChild.nodeValue = filename
    file.getElementsByTagName( "name" )[0].firstChild.nodeValue = object_name

    with open(path_save, "w") as fs: 
        fs.write( file.toxml() )
        fs.close() 


def image_is_label(in_dir, object_name):
    # This function will rename your images then create .xml files that contain the annotations.
    
    images = glob(os.path.join(in_dir, '*'))
    for i, image_ in enumerate(images):
        image_name_ = object_name+'_'+str(i)
        if not os.path.exists(os.path.join(in_dir, image_name_ +'.jpg')):
            os.rename(image_, os.path.join(in_dir, image_name_ +'.jpg'))

        image = Image.open(os.path.join(in_dir, image_name_ +'.jpg'))
        width = image.width
        height = image.height
        depth = len(image.split())
        xmin = 0
        ymin = 0
        xmax = width - 1
        ymax = height - 1

        fill_xml(os.path.join(in_dir, image_name_ +'.xml'), width, height, depth, xmin, ymin, xmax, ymax, image_, image_name_+'.jpg', object_name)

    print('Files saved successfully')


def create_synthetic(bg_images, fg_image, object_name, file_name='image', save_path='synthetic_data', iterations=10):  
    # This function is to put image over image and it saves the xml file for the annotations.
    # bg_image: path to all the background images.
    # fg_image: path to the forground image (in our case, it is an image of a shadow)
    # iterations: the number of the synthetic data that you want to create
    
    #image_list = glob('C:/Users/amine/Documents/Amine_Files/PhD/codes/images/img_skyhero_resized_rgb/*')
    image_list = glob(os.path.join(bg_images, '*'))
    shadow_image = Image.open(fg_image)
    #path_save = 'C:/Users/amine/Documents/Amine_Files/PhD/codes/Yolov3_Shadow_mixed_data/data/sky_hero_synthetic_data_with_xml/test'
    if not os.path.exists(save_path): os.mkdir(save_path)
    blur_list = [True, False]

    for i in tqdm(range(iterations)):
        E = False
        # Random positions
        position = (random.randint(-100, 1000), random.randint(-100, 1000))

        # Random blur
        blur_index = random.randint(0, 1)
        blur = blur_list[blur_index]

        # Randrom opcity values from the defined list
        list_rand = [0.05,0.4,0.08,0.3,0.1]
        n = random.randint(0, len(list_rand)-1)
        opacity = list_rand[n]

        # Random resizing values
        resize_value = random.randint(0,200)

        # Choose a random image from the choosen grayscale ones
        image_index = random.randint(0, len(image_list) - 1)
        background_ = image_list[image_index]
        background = image_list[image_index]
        background = Image.open(background).convert('RGB')

        # Return the dimensions of the background image so that we define the position of the bbox
        width = background.width
        height = background.height
        depth = len(background.split())

        # Case 1
        xmin = position[0]
        ymin = position[1]
        xmax = xmin + shadow_image.width - resize_value
        ymax = ymin + shadow_image.height - resize_value

        # Case 2
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > width: xmax = width-1
        if ymax > height: ymax = height-1

        # Case 3
        if xmin >= width or ymin >= height or xmax <= 0 or ymax <= 0: E = True

        # If there is no error then save the image + the label otherwise it won't save anything
        if not E:
            image_over_image(fg_image, background_,
                            save_path=os.path.join(save_path, file_name+str(i)+'.jpg'),
                            resize_value = resize_value,
                            opacity=opacity,
                            position=position,
                            blur=blur,
                            save=True)

            fill_xml(os.path.join(save_path, file_name+str(i)+'.xml'), width, height, depth,
                     xmin, ymin, xmax, ymax,
                     path_image=os.path.join(save_path, file_name+str(i)+'.jpg'),
                     filename=file_name+str(i)+'.jpg',
                     object_name=object_name)

    print('You files are saved in: ', save_path)


def edit_xml(in_dir, out_dir, object_name, file_name, copy_image=True):
    file = md.parse(in_dir) 
    if not os.path.join(out_dir): os.mkdir(out_dir)

    #file.getElementsByTagName( "path" )[0].firstChild.nodeValue = os.path.join(out_dir, file_name + '.jpg')
    file.getElementsByTagName( "filename" )[0].firstChild.nodeValue = file_name + '.jpg'
    
    for i in range(len(file.getElementsByTagName( "name" ))):
        file.getElementsByTagName( "name" )[i].firstChild.nodeValue = object_name

    with open(os.path.join(out_dir, file_name + '.xml'), "w") as fs: 
        fs.write( file.toxml() )
        fs.close() 
    
    if copy_image:
        if not os.path.exists(os.path.join(out_dir, file_name)): 
            os.rename(in_dir.replace('.xml', '.jpg'), os.path.join(out_dir, file_name + '.jpg'))


