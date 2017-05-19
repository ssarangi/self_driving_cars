import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
from sklearn.externals import joblib
import glob

from vehicle_detection import *

def view_results(model):
    files = list(glob.iglob("cropped/*.png"))

    cols = 4
    rows = len(files) // 5 + 1

    fig, axis = plt.subplots(rows, cols, figsize=(20, 20))
    spatial_size = (16, 16)
    for i, f in enumerate(files):
        img = mpimg.imread(f)
        row = i // 5
        cropped = cv2.resize(img, spatial_size)
        features, hog_images = extract_feature_from_image(cropped, spatial_size)
        features = create_final_feature_vector(features)
        res = model.predict(features)[0]
        if res == 1:
            print(f)
        axis[row, 0].imshow(img)
        txt = "Car" if res == 1 else "Not Car"
        axis[row, 0].set_title(txt)
        axis[row, 0].axis('off')
        i = 1
        for hog_image in hog_images:
            axis[row, i].imshow(hog_image, cmap='gray')
            axis[row, i].axis('off')
            i += 1

    plt.savefig('images.png', bbox_inches='tight')

def main():
    model = joblib.load('model.pkl')
    view_results(model)

if __name__ == "__main__":
    main()