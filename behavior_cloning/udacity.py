from driverless import *

def display_change_brightness_graph(images, steering_angles):
    plt.subplots(figsize=(20, 5))
    for i, img in enumerate(images):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.title("Steering: {:.4f}".format(steering_angles[i]))
        plt.imshow(img)
    plt.show()

def main():
    df = read_training_data("track1")
    df = rearrange_and_augment_dataframe(df, shuffle_data=True)

    # Get the first 10 images
    df_first_10 = df[1:11]

    images = []
    angles = []
    for i, row in df_first_10.iterrows():
        img = read_image(row.image)
        steering_angle = row.steering_angle
        img = change_brightness(img)
        images.append(img)
        angles.append(steering_angle)

    display_change_brightness_graph(images, angles)

if __name__ == "__main__":
    main()
