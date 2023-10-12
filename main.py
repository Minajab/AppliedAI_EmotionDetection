import os
from PIL import Image, ImageFilter
import cv2


def Gray_scale(Input_directory):
    os.chdir(Input_directory)
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Image = cv2.imread(Path)
            if len(Image.shape) != 2:
                Gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(Path, Gray_image)


def Size_determination(Input_directory):
    os.chdir(Input_directory)
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


def Contrast_regulation(Input_directory):
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Image = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
            Equalized_image = cv2.equalizeHist(Image)
            cv2.imwrite(Path, Equalized_image)


def Resize_image(Input_directory, Target_width, Target_height):
    os.chdir(Input_directory)
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Img = Image.open(Path)
            Resized = Img.resize((Target_width, Target_height), Image.LANCZOS)
            sharpened_image = Resized.filter(ImageFilter.SHARPEN)
            sharpened_image.save(Path)


def Denoise(Input_directory):
    os.chdir(Input_directory)
    for Class in ["Angry", "Bored", "Focused", "Neutral"]:
        Files = os.listdir(Class)
        for File in Files:
            Path = os.path.join(Class, File)
            Image = cv2.imread(Path)
            Denoised_gaussian = cv2.GaussianBlur(Image, (5, 5), 0)
            cv2.imwrite(Path, Denoised_gaussian)


def main():
    Input_directory = "C:\\Users\\mahshad\\Desktop\\AI\\Data\\"
    Gray_scale(Input_directory)
    [Width, Height] = Size_determination(Input_directory)
    Contrast_regulation(Input_directory)
    Resize_image(Input_directory, Width, Height)
    Denoise(Input_directory)


main()
