import numpy as np
import cv2
import imutils
import PIL
from PIL import Image, ImageDraw, ImageFilter
from argparse import ArgumentParser
import glob

def parse_args():
    """Command-line argument for rain synthesis"""

    # New parser
    parser = ArgumentParser(description='Rain Synthesis')

    # Data parameters
    # rain params
    parser.add_argument('-ks', '--kernel-size',
                        help='kenel size of motion blur, can be used to increase rain streak size', default='30', type=int)
    parser.add_argument('-dl', '--drop-length', help='rain streak length. best to keep it small', default=10, type=int)
    parser.add_argument('-dw', '--drop-width', help='thickness of streak', default=1, type=int)
    parser.add_argument('-deg', '--degree', help='angle of rain streak', default=33, type=int)
    parser.add_argument('-s', '--slant', help='slant of streak. (don\'t use since we using rotate)', default=0, type=int) 
    parser.add_argument('-c', '--color', help='rain streak color', default=200, type=int)
    parser.add_argument('-rc', '--rain-count', help='rain drop count', default=20, type=int)
    
    # dataset
    parser.add_argument('-dgt', '--data-gt', help='data ground truth root path', default='ground-truth', type=str)
    parser.add_argument('-dt', '--data-type', help='data img type', default='jpg')
    
    # params for creating rain variance dataset
    parser.add_argument('-rlr', '--rain-lower-range', help='how many different rain types to produce, lower range', default=20, type=int)
    parser.add_argument('-rur', '--rain-upper-range', help='how many different rain types to produce, upper range', default=101, type=int)
    parser.add_argument('-ri', '--rain-increment', help='how much to increment every loop. range(rlr, rur, ri)', default=30, type=int)
    parser.add_argument('-r', '--rain-image-count', help='how many different rain images per rain type', default=100, type=int)

    return parser.parse_args()


def motion_blur(blur_img, kernel_size=30):

    # Specify the kernel size.
    # The greater the size, the more the motion.
    
    # Create the vertical kernel.
    kernel_v = np.zeros((int(kernel_size), int(kernel_size)))

    # Fill the middle row with ones.
    kernel_v[:, int((int(kernel_size) - 1)/2)] = np.ones(int(kernel_size))

    # Normalize.
    kernel_v /= int(kernel_size)

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(blur_img, -1, kernel_v)

    return vertical_mb


def rotate(img, deg=33):
    rotated = imutils.rotate_bound(img, deg)
    return rotated


def rotate_and_blur(img, deg=33, blur_kernel=30):
    originalX = img.shape[0]
    originalY = img.shape[1]

    x = (max(originalX,originalY) * 2) // 1
    y = (max(originalX,originalY) * 2) // 1
    
    
    print(max(originalX,originalY), originalX, originalY, x, y)
    # rotate blur and rotate back
    blured_img = motion_blur(img, kernel_size = blur_kernel)
    rotated = imutils.rotate_bound(blured_img, deg)

    # crop
    x_hat = rotated.shape[0]
    y_hat = rotated.shape[1]

    dx = x_hat - originalX
    dy = y_hat - originalY

    if dx % 2 == 1:
        x_hat -= 1
        dx -= 1
    if dy % 2 == 1:
        y_hat -= 1
        dy -= 1

    # shape = (x, y)
    rotated = rotated[dx//2 : x_hat - (dx//2), dy//2 : y_hat - (dy//2)]

    # shape = (x, y, 1)
    rotated = np.expand_dims(rotated[:,:,0], axis=2)
    return rotated


def generate_random_lines(imshape,slant,drop_length,rain_count):
    drops=[]
#     print(imshape[0], imshape[1] * 2)
    imshape[0] =  imshape[0] * 2
    imshape[1] = imshape[1] * 2
    
    ## If You want heavy rain, try increasing rain_count
    for i in range(rain_count):
        if slant<0:
            x = np.random.randint(slant,imshape[1])
            y = np.random.randint(drop_length, imshape[0])
        else:
            x = np.random.randint(0,imshape[1]-slant)
            y = np.random.randint(0,imshape[0]-drop_length)

        drops.append((x,y))
    return drops

def add_rain(image, drop_length=10, slant=0, drop_width=1, drop_color=(200,200,200), rain_count=20):
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

    rain_streak_img = np.zeros((imshape[0], imshape[1],imshape[2]))

    rain_drops = generate_random_lines(imshape,slant,drop_length, rain_count)
    for rain_drop in rain_drops:
        cv2.line(rain_streak_img,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)

    rain_streak_img = rotate_and_blur(rain_streak_img, deg = args.degree, blur_kernel=args.kernel_size)
    return rain_streak_img

if __name__ == '__main__':
    # Parse test parameters
    args = parse_args()
    
    img_path = f"img/image.jpg"
    img = cv2.imread(img_path)
    
    for i in range(20, 101, 10):
        rainy_img = add_rain(img, rain_count=i)
        cv2.imwrite(f'testest/just_rain{i}.jpg',  rainy_img)

        cv2.imwrite(f'testest/rainy_img{i}.jpg',  img + rainy_img)

    
        
    
#     images_dataset = glob.glob(f'./{args.data_gt}/*.{args.data_type}')
    
#     lower_range = args.rain_lower_range
#     upper_range = args.rain_upper_range
#     increment = args.rain_increment
#     img_n = args.rain_image_count

#     rain_types = (upper_range - lower_range) // increment + 1
    
#     # TRAIN
#     for idx, image in enumerate(images_dataset[:int(len(images_dataset) * 0.6 )]):
#         img = cv2.imread(f"{image}")

#         # 3 rain types
#         # 100 per type = 500 rainy images per clean image
#         for rain_count in range(lower_range, upper_range, increment):
#             for i in range(img_n):
#                 rainy_img = add_rain(img, drop_length=args.drop_length, slant=args.slant, drop_width=args.drop_width, drop_color=(args.color,args.color,args.color), rain_count=rain_count)
#                 cv2.imwrite(f'train/img_n{idx}_rc{rain_count}_{i}.jpg',  img + rainy_img)

#     # VALIDATION
#     for idx, image in enumerate(images_dataset[int(len(images_dataset) * 0.6) : int(len(images_dataset) * 0.8)]):
#         img = cv2.imread(f"{image}")
#         idx += img_n * (int(len(images_dataset) * 0.6) * rain_types # for naming sake

#         # 3 rain types
#         # 100 per type = 500 rainy images per clean image
#         for rain_count in range(lower_range, upper_range, increment):
#             for i in range(img_n):
#                 rainy_img = add_rain(img, drop_length=args.drop_length, slant=args.slant, drop_width=args.drop_width, drop_color=(args.color,args.color,args.color), rain_count=rain_count)
#                 cv2.imwrite(f'validation/img_n{idx}_rc{rain_count}_{i}.jpg', img + rainy_img)

#     # TEST
#     for idx, image in enumerate(images_dataset[int(len(images_dataset) * 0.8): ]):
#         img = cv2.imread(f"{image}")
#         idx = img_n * (int(len(images_dataset) * 0.8)) * rain_types # for naming sake

#         # 3 rain types
#         # 100 per type = 500 rainy images per clean image
#         for rain_count in range(lower_range, upper_range, increment):
#             for i in range(img_n):
#                 rainy_img = add_rain(img, drop_length=args.drop_length, slant=args.slant, drop_width=args.drop_width, drop_color=(args.color,args.color,args.color), rain_count=rain_count)
#                 cv2.imwrite(f'test/img_n{idx}_rc{rain_count}_{i}.jpg', img + rainy_img)
