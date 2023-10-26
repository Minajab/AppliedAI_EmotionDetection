import os
from PIL import Image, ImageFilter, ImageEnhance, ImageFilter
import cv2
import numpy as np
from skimage import io
import random

#Extracts faces and removes backgrounds for 'bored' and 'focused' classes.
def Deleting_background():
    for Class in ["Bored", "Focused"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            img = cv2.imread(Path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('C:\\Users\\mahshad\\Desktop\\haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            print('Number of detected faces:', len(faces))
            counter = 0
            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    face = img[y:y + h, x:x + w]
                    new_apth = Path.replace(".", "_" + str(counter) + ".")
                    cv2.imwrite(new_apth, face)
                    print(f"face{i}.jpg is saved")
                counter += 1
                os.remove(Path)

#Applies slight rotation to 'angry' and 'neutral' class images; other classes already augmented.
def Rotation():
    min_rotation = -30.0
    max_rotation = 30.0
    for Class in ["Angry", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            image = cv2.imread(Path)
            random_number = random.uniform(0, 1)
            if (random_number < 0.5):
                rotation_angle = random.uniform(min_rotation, max_rotation)
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
                cv2.imwrite(Path, rotated_image)

#Applied to 'focused' and 'bored' classes with a saturation factor of 1.5 for color enhancement.
def Saturation_adjustment():
    for Class in ["Focused", "Bored"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Img = Image.open(Path)
            if (Img.mode != "L"):
                Enhancer = ImageEnhance.Color(Img)
                Saturation_factor = 1.5
                Img_with_adjusted_saturation = Enhancer.enhance(Saturation_factor)
                Img_with_adjusted_saturation.save(Path)
                Img.close()
                Img_with_adjusted_saturation.close()
                print("one saturation is done")

#This function converts the images to grayscale.
def Gray_scale():
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Image = cv2.imread(Path)
            if len(Image.shape) != 2:
                Gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(Path, Gray_image)
                print("gray")

#Finds maximum image width and height, then sets all images to match this size using 'Resize_image'.
def Size_determination():
    Classes = ["Angry", "Bored", "Focused", "Neutral"]
    Max_width = 0
    Max_height = 0
    for Class in Classes:
        Files = os.listdir(Class)
        for File in Files:
            Img = Image.open((os.path.join(Class, File)))
            Width, Height = Img.size
            if Width > Max_width:
                Max_width = Width
            if Height > Max_height:
                Max_height = Height
    return [Max_width, Max_height]

#Adjusts the contrast with an intensity factor of 1.1 after converting the images to grayscale.
def Contrast_regulation():
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            img = Image.open(Path)
            enhancer = ImageEnhance.Contrast(img)
            contrast_factor = 1.1
            img_with_adjusted_contrast = enhancer.enhance(contrast_factor)
            img_with_adjusted_contrast.save(Path)
            img.close()
            img_with_adjusted_contrast.close()
            print("contrast regulation")


def Resize_image(Target_width, Target_height):
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        print("class")
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Img = Image.open(Path)
            Resized = Img.resize((Target_width, Target_height), Image.LANCZOS)
            sharpened_image = Resized.filter(ImageFilter.SHARPEN)
            sharpened_image.save(Path)
            print("resized")

#Applies fastNlMeansDenoising algorithm to remove noise and preserve image details using similar patches.
def Denoise():
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Img = cv2.imread(Path)
            Dst = cv2.fastNlMeansDenoising(Img, None, 15, 7, 21)
            cv2.imwrite(Path, Dst)
            print("denoise")

#This function applies a two-step blurring process: bilateral filter for edge-preserving smoothing, followed by a 3x3 Gaussian blur for further refinement.
def Deblock():
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        print(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Image = cv2.imread(Path)
            Deblocked_image = cv2.bilateralFilter(Image, d=9, sigmaColor=20, sigmaSpace=20)
            Blurred_image = cv2.GaussianBlur(Deblocked_image, (3, 3), 0)
            cv2.imwrite(Path, Blurred_image)
            print("deblocked")

#Mediating the impact of block artifacts through a chain of erosion and dilation.
def morphology():
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            print(Path)
            gray_image = cv2.imread(Path)
            kernel_size = (2, 2)
            dilation_kernel = np.ones(kernel_size, np.uint8)
            erosion_kernel = np.ones(kernel_size, np.uint8)
            dilated_image = cv2.dilate(gray_image, dilation_kernel, iterations=1)
            eroded_image = cv2.erode(dilated_image, erosion_kernel, iterations=1)
            cv2.imwrite(Path, eroded_image)

#The main function calls all processing functions in a specific order.
def main():
    Input_directory = "C:\\Users\\mahshad\\Desktop\\AI\\Data\\"
    os.chdir(Input_directory)
    Deleting_background()
    Rotation()
    Saturation_adjustment()
    Gray_scale()
    Contrast_regulation()
    morphology()
    Denoise()
    Deblock()
    [Width, Height] = Size_determination()
    Resize_image(Width, Height)


main()
