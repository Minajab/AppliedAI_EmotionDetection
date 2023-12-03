import os
import random
from PIL import Image


def rotate_image(current_directory, input_path, angle, counter):
    print(input_path)
    image = Image.open(input_path)
    rotated_image = image.rotate(angle)
    rotated_image.save(current_directory + "\\" + str(counter) + "_new.jpg")


def flip_image(current_directory, input_path, counter):
    image = Image.open(input_path)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image.save(current_directory + "\\" + str(counter) + "_new.jpg")


def zoom_in(current_directory, image_path, zoom_factor, counter):
    image = Image.open(image_path)
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    zoomed_image = image.crop((left, top, right, bottom))
    zoomed_image.save(current_directory + "\\" + str(counter) + "_new.jpg")


def shift_image(current_directory, image_path, shift_width, shift_height, counter):
    image = Image.open(image_path)
    width, height = image.size
    new_width = width - 2 * shift_width
    new_height = height - 2 * shift_height
    box = (shift_width, shift_height, width - shift_width, height - shift_height)
    shifted_image = image.crop(box)
    shifted_image.save(current_directory + "\\" + str(counter) + "_new.jpg")


counter = 0
parent_directory = 'C:\\Users\\mahshad\\Desktop\\2\\'
os.chdir(parent_directory)

subdirectories = [d for d in os.listdir() if os.path.isdir(os.path.join(parent_directory, d))]

for subdirectory in subdirectories:
    current_directory = os.path.join(parent_directory, subdirectory)
    os.chdir(current_directory)
    files_in_current_directory = os.listdir()
    if len(files_in_current_directory) < 50:
        t = len(files_in_current_directory)
        while (t < 50):
            file=files_in_current_directory[random.randint(0, len(files_in_current_directory)-1)]
            random_number = random.randint(0, 3)
            random_number = 0
            if (random_number == 0):
                rotate_image(current_directory, file, random.randint(-30, 30), counter)
            if (random_number == 1):
                flip_image(current_directory, file, counter=counter)
            if (random_number == 2):
                zoom_in(current_directory, file, random.random(0, 3), counter)
            if (random_number == 3):
                shift_image(current_directory, file, random.random(1, 2), counter)
            counter += 1
            t +=1
