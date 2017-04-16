from model import *

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

def display_original_image():
    cols = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'brake', 'speed']
    df = pd.read_csv('data/track1_1/driving_log.csv', names=cols)
    # Get 3 rows of data and display them
    print(len(df))
    idx = np.random.randint(1000, size=3)
    print(idx)
    ndf = df.ix[idx]

    fig, axs = plt.subplots(3, 3, figsize=(5, 5))
    i = 0
    for idx, row in ndf.iterrows():
        axs[i][0].imshow(read_image(row['left_image']))
        axs[i][1].imshow(read_image(row['center_image']))
        axs[i][2].imshow(read_image(row['right_image']))

        steering_angle = float(row['steering_angle'])
        axs[i][0].set_title("Left Image Angle: {:.4f}".format(steering_angle + 0.25))
        axs[i][1].set_title("Center Image Angle: {:.4f}".format(steering_angle))
        axs[i][2].set_title("Right Image Angle: {:.4f}".format(steering_angle - 0.25))

        axs[i][0].axis('off')
        axs[i][1].axis('off')
        axs[i][2].axis('off')

        axs[i][0].set_axis_bgcolor('gray')
        axs[i][1].set_axis_bgcolor('gray')
        axs[i][2].set_axis_bgcolor('gray')
        i += 1

    fig.tight_layout()
    plt.show()


def main():
    df = read_training_data("track1")
    df = rearrange_and_augment_dataframe(df, shuffle_data=True)

    # Get the first 10 images
    # df_first_10 = df[1:11]
    #
    # images = []
    # angles = []
    # for i, row in df_first_10.iterrows():
    #     img = read_image(row.image)
    #     steering_angle = row.steering_angle
    #     img = change_brightness(img)
    #     images.append(img)
    #     angles.append(steering_angle)
    #
    # # display_change_brightness_graph(images, angles)
    #
    # df_10 = df[20:25]
    # images = []
    # angles = []
    # for i, row in df_10.iterrows():
    #     img = read_image(row.image)
    #     steering_angle = row.steering_angle
    #     images.append(img)
    #     angles.append(steering_angle)
    #
    # display_flipped_image(images, angles)
    display_original_image()

if __name__ == "__main__":
    main()
