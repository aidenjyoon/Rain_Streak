import os
import numpy as np
import cv2
import imutils
import PIL
from PIL import Image, ImageDraw, ImageFilter
from argparse import ArgumentParser
import glob

parser = ArgumentParser(description='Rain Synthesis')

# Data parameters
# rain params
parser.add_argument('--kernel-size',
                    help='kenel size of motion blur, can be used to increase rain streak size', default='30', type=int)
parser.add_argument('--drop-length', help='rain streak length. best to keep it small', default=10, type=int)
parser.add_argument('--drop-width', help='thickness of streak', default=1, type=int)
parser.add_argument('--degree', help='angle of rain streak', default=33, type=int)
parser.add_argument('--slant', help='slant of streak. (don\'t use since we using imutil.rotate)', default=0, type=int) 
parser.add_argument('--color', help='rain streak color', default=200, type=int)
# parser.add_argument('-rc', '--rain-count', help='rain drop count', default=20, type=int)

# dataset
parser.add_argument('--data-gt', help='data ground truth root path', default='ground-truth', type=str, required=True)
parser.add_argument('--data-type', help='data img type', default='jpg')

# params for creating rain variance dataset
parser.add_argument('--rain-lower-range', help='how many different rain types to produce, lower range', default=200, type=int)
parser.add_argument('--rain-upper-range', help='how many different rain types to produce, upper range', default=1001, type=int)
parser.add_argument('--rain-increment', help='how much to increment every loop. range(rlr, rur, ri)', default=300, type=int)
parser.add_argument('--rain-images-count', help='how many different rain images per rain type', default=100, type=int)



def motion_blur(blur_img, kernel_size=30):
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = args.kernel_size

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(blur_img, -1, kernel_v)
    
    return vertical_mb


def rotate(img, deg=33):
    deg = args.degree
    
    rotated = imutils.rotate_bound(img, deg)
    return rotated

def rotate_and_blur(img, deg=33, blur_kernel=30):
    deg = args.degree
    blur_kernel = args.kernel_size

    x = img.shape[0]
    y = img.shape[1]
    
    # enlarge to fit rain in image even if we rotate
    x_enlarged = img.shape[1] * 2
    y_enlarged = img.shape[0] * 2
    
    dsize = (x_enlarged, y_enlarged)
             
    img_enlarged = cv2.resize(img, dsize)

    # blur and rotate to desired angle
    blured_img = motion_blur(img_enlarged, kernel_size = blur_kernel)
    rotated_img = rotate(blured_img, deg)
    
    # crop
    x_hat = rotated_img.shape[0]
    y_hat = rotated_img.shape[1]
    
    dx = x_hat - x
    dy = y_hat - y
    
    # if odd make pixel size even
    if dx % 2 == 1:
        x_hat -= 1
        dx -= 1
    if dy % 2 == 1:
        y_hat -= 1
        dy -= 1
        

    rotated_img = rotated_img[dx//2 : x_hat - (dx//2), dy//2 : y_hat - (dy//2)]

    rotated_img = np.expand_dims(rotated_img[:,:,0], axis=2)
    return rotated_img

def generate_random_lines(imshape, slant, drop_length, rain_count):
    drops=[]
    
    ## If You want heavy rain, try increasing rain_count
    for i in range(rain_count):
        if slant<0:
            x = np.random.randint(slant, imshape[1])
            y = np.random.randint(drop_length, imshape[0])
        else:
            x = np.random.randint(0,imshape[1]-slant)
            y = np.random.randint(0,imshape[0]-drop_length)

        drops.append((x,y))
    return drops

# for drop_color
c = 200

def add_rain(image, slant=0, drop_length=15, drop_width=1, drop_color=(c,c,c), rain_count=100):
    '''
    PARAMS:
    image - input image
    drop_length - how long rain streaks are
    slant - slant angle
    drop_width - rain streak thickness
    drop_color - color
    rain_count - number of droplets
    '''
    imshape = image.shape
    
    # to have distant effect with darker rain color
    alpha = 0.2
    background_rain_color = (c * alpha, c * alpha, c * alpha)
    
    # rain streaks 1
    rain_streak_img1 = np.zeros((imshape[0], imshape[1],imshape[2]))
    rain_drops = generate_random_lines(imshape,slant,drop_length, rain_count)

    for rain_drop in rain_drops:
        cv2.line(rain_streak_img1,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
        
    # rain streaks 2 - darker rain with shorter drop_length to create distance effect
    rain_streak_img2 = np.zeros((imshape[0], imshape[1],imshape[2]))
    rain_drops = generate_random_lines(imshape,slant,drop_length, rain_count)

    for rain_drop in rain_drops:
        cv2.line(rain_streak_img2,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+int(drop_length*0.8)),background_rain_color,drop_width)
        
    rain_streak_img1 = rotate_and_blur(rain_streak_img1, blur_kernel=30)
    rain_streak_img2 = rotate_and_blur(rain_streak_img2, blur_kernel=30)
    
    rain_streak_img = rain_streak_img1 + rain_streak_img2
    return rain_streak_img

if __name__ == '__main__':
    args = parser.parse_args()

    images_dataset = glob.glob(f'./{args.data_gt}/*.{args.data_type}')

    lower_range = args.rain_lower_range
    upper_range = args.rain_upper_range
    increment = args.rain_increment
    img_n = args.rain_images_count

    rain_types = (upper_range - lower_range) // increment + 1

    # check if directories exist if not create
    train_path = './train'
    validation_path = './validation'
    test_path = './test'
    isTrain_dir = os.path.isdir(train_path)
    isVal_dir = os.path.isdir(validation_path)
    isTest_dir = os.path.isdir(test_path)
    
    if isTrain_dir == False:
        os.mkdir(train_path)
        print(f'made directory {train_path}')
    if isVal_dir == False:
        os.mkdir(validation_path)
        print(f'made directory {validation_path}')

    if isTest_dir == False:
        os.mkdir(test_path)
        print(f'made directory {test_path}')

    # TRAIN
    for idx, image in enumerate(images_dataset[:int(len(images_dataset) * 0.6 )]):
        print('making TRAINING rain images...')
        
        img = cv2.imread(f"{image}")
        
        slant = args.slant
        drop_length = args.drop_length
        drop_width = args.drop_width
        drop_color = (args.color, args.color, args.color)
        
        # 3 rain types
        # 100 per type = 500 rainy images per clean image
        for rain_count in range(lower_range, upper_range+1, increment):
            for i in range(img_n):
                rainy_img = add_rain(img, slant, drop_length, drop_width, drop_color, rain_count)
                cv2.imwrite(f'train/img_n{idx}_rc{rain_count}_{i}.jpg',  img + rainy_img)

    # VALIDATION
    for idx, image in enumerate(images_dataset[int(len(images_dataset) * 0.6) : int(len(images_dataset) * 0.8)]):
        print('making VALIDATION rain images...')

        img = cv2.imread(f"{image}")
        idx += img_n * (int(len(images_dataset) * 0.6) * rain_types) # for naming sake

        slant = args.slant
        drop_length = args.drop_length
        drop_width = args.drop_width
        drop_color = (args.c, args.c, args.c)
        
        # 3 rain types
        # 100 per type = 500 rainy images per clean image
        for rain_count in range(lower_range, upper_range+1, increment):
            for i in range(img_n):
                rainy_img = add_rain(img, slant, drop_length, drop_width, drop_color, rain_count)
                cv2.imwrite(f'validation/img_n{idx}_rc{rain_count}_{i}.jpg', img + rainy_img)

    # TEST
    for idx, image in enumerate(images_dataset[int(len(images_dataset) * 0.8): ]):
        print('making TEST rain images...')
        
        img = cv2.imread(f"{image}")
        idx = img_n * (int(len(images_dataset) * 0.8)) * rain_types # for naming sake

        slant = args.slant
        drop_length = args.drop_length
        drop_width = args.drop_width
        drop_color = (args.c, args.c, args.c)
        
        # 3 rain types
        # 100 per type = 500 rainy images per clean image
        for rain_count in range(lower_range, upper_range+1, increment):
            for i in range(img_n):
                rainy_img = add_rain(img, slant, drop_length, drop_width, drop_color, rain_count)
                cv2.imwrite(f'test/img_n{idx}_rc{rain_count}_{i}.jpg', img + rainy_img)
