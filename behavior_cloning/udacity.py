from driverless import *

def display_change_brightness_graph(images, steering_angles):
    plt.subplots(figsize=(20, 5))
    for i, img in enumerate(images):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.title("Steering: {:.4f}".format(steering_angles[i]))
        plt.imshow(img)
    plt.show()

def display_flipped_image(images, steering_angles):
    total_images = len(images)
    fig, ax = plt.subplots(nrows=total_images, ncols=2, figsize=(5, 8))
    for i, img in enumerate(images):
        ax[i][0].axis('off')
        ax[i][0].imshow(img)
        ax[i][0].set_title("Original Angle: {:.4f}".format(steering_angles[i]))

        ax[i][1].axis('off')
        ax[i][1].set_title("Flipped Angle: {:.4f}".format(-1.0 * steering_angles[i]))
        ax[i][1].imshow(flip_image(img))
    fig.tight_layout()
    plt.show()

def display_sample_images(df):
    

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

    # display_change_brightness_graph(images, angles)

    df_10 = df[20:25]
    images = []
    angles = []
    for i, row in df_10.iterrows():
        img = read_image(row.image)
        steering_angle = row.steering_angle
        images.append(img)
        angles.append(steering_angle)

    display_flipped_image(images, angles)

if __name__ == "__main__":
    main()
