from __future__ import print_function

import linecache
import os
import random


class ImageGenerator():
    def __init__(self, img_dir, img_class_map_file, class_file):

        # Init params
        self.image_dir = img_dir
        self.img_class_map_file = img_class_map_file
        self.line_pointer = 1

        # Read in the class labels
        self.class_names = list()
        with open(class_file, 'r') as f:
            for line in f:
                self.class_names.append(" ".join(line.split()[1:]))

        # Count the number of images available
        with open(img_class_map_file) as f:
            self.n_images = sum(1 for _ in f)

    def __iter__(self):
        return self

    def next(self):
        line = linecache.getline(self.img_class_map_file, self.line_pointer).split()
        self.line_pointer += 1

        if line == '':
            raise StopIteration

        return line

    def path_and_class_generator(self, n=None, img_nums=None):
        """
        Generates n random image paths and their corresponding class label

        Args:
            n (int): The number of items to generate

        Yields:
            string: The path to the image
            string: The ground truth label of the image

        """
        if n > self.n_images:
            raise ValueError("n is larger than the number of images available")

        if n is None:
            n = self.n_images

        # Generate n unique random line numbers if img_nums is not set
        if not img_nums:
            line_nums = random.sample(range(1, self.n_images+1), n)
        else:
            line_nums = img_nums

        for line in line_nums:
            image_path, class_num = linecache.getline(self.img_class_map_file, line).split()
            image_path = os.path.join(self.image_dir, image_path)
            image_label = self.class_names[int(class_num)]
            yield (image_path, image_label)

    def get_image_data(self, img_num):
        """
        Return the image path and the correct image label for the given img_num
        """
        image_path, class_num = linecache.getline(self.img_class_map_file, img_num).split()
        image_path = os.path.join(self.image_dir, image_path)
        image_label = self.class_names[int(class_num)]
        return image_path, image_label
    
    def get_image_filename(self, img_num):
        """
        Return the image filename for the given img_num
        """
        image_filename, class_num = linecache.getline(self.img_class_map_file, img_num).split()
        #image_path = os.path.join(self.image_dir, image_path)
        #image_label = self.class_names[int(class_num)]
        return image_filename


def main():
    val_generator = ImageGenerator('val/images', 'val/val.txt', 'val/synset_words.txt')

    for path, img_class in val_generator.path_and_class_generator(10):
        print(path, img_class)


if __name__ == '__main__':
    main()
