import os
import argparse
from PIL import Image


def create_depth_edges_gt_images(directory_path, H, W):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Checks if the file is a text file
            filepath = os.path.join(directory_path, filename)
            img = Image.new('1', (W, H), 0)  # Create a new black image
            pixels = img.load()

            with open(filepath, 'r') as file:
                for line in file:
                    y, x = map(int, line.strip().split(' '))
                    if 0 <= x < W and 0 <= y < H:
                        pixels[x, y] = 1  # Set pixel to white

            img.save(filepath.replace(".txt", ".png"))  # Save the image as PNG
    print("Dataset loaded from " + directory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from coordinate text files.')
    parser.add_argument('--directory_path', type=str, help='Directory path containing the gt tet files')
    parser.add_argument('--height', type=int, default=384, help='Height of the images')
    parser.add_argument('--width', type=int, default=1280, help='Width of the images')

    args = parser.parse_args()

    create_depth_edges_gt_images(args.directory_path, args.height, args.width)
