import numpy as np
import cv2, os
import sys
import pandas as pd

class dl_model():

    EXT_TRAIN_DATA = 'train'
    EXT_TEST_DATA = 'test'
    EXT_TRAIN_CSV = 'trainLabels.csv'
    
    IMAGE_WIDTH = 1536
    IMAGE_HEIGHT = 1024
    N_CHANNELS = 3
    
    BATCH_SIZE = 10
    
    def __init__(self, argv):
        self.argv = argv
        self.BASE_PATH = argv[0]

    def get_image_name_list(self, path, train_or_not):
        if train_or_not:
            training_csv = pd.read_csv(path)
            headers = training_csv.columns
            return np.array([training_csv[headers[0]], training_csv[headers[1]]])
        else:
            return np.array([os.listdir(path)])


    def get_image_names(self):
        self.train_image_names_with_labels = self.get_image_name_list(os.path.join(self.BASE_PATH, self.EXT_TRAIN_CSV), 1) # returns a tuple
        self.test_image_names = self.get_image_name_list(os.path.join(self.BASE_PATH, self.EXT_TEST_DATA), 0) # returns just names

        print('Number of training images: {}\nNumber of testing images: {}'.format(len(self.train_image_names_with_labels[0]), len(self.test_image_names[0])))


    def image_transformation(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        return np.array(img).reshape((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.N_CHANNELS))


    def image_batch_generator(self, array, batch_size, ext):
        path = os.path.join(self.BASE_PATH, ext)
        for i in range(0, len(array[0]), batch_size):
            batch = array[0][i: i+batch_size]
            data_batch = []
            for j, image_name in enumerate(batch):
                try:
                    if ext == self.EXT_TRAIN_DATA:
                        image_path = '{}.jpeg'.format(os.path.join(path, image_name))
                        data_batch.append((self.image_transformation(image_path), array[1][i+j]))
                    else:
                        image_path = '{}'.format(os.path.join(path, image_name))
                        data_batch.append(self.image_transformation(image_path))
                except:
                    print('Error reading: {}'.format(image_path))
            yield(data_batch)
    
    
    def execute(self):
        self.get_image_names()
        training_batch_generator = self.image_batch_generator(self.train_image_names_with_labels, self.BATCH_SIZE, self.EXT_TRAIN_DATA)


#        test_batch_generator = self.image_batch_generator(self.test_image_names, self.BATCH_SIZE, self.EXT_TEST_DATA)


if __name__ == '__main__':
    dl_model(sys.argv[1:]).execute()
