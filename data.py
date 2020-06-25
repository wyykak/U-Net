from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob

#flexible data converter and loader
class dataProcess(object):
    #data_path - training set raw image path
    #label_path - training set split image path
    #test_path - testing set raw image path
    #npy_path - converted numpy array file path
    #img_type - file type
    def __init__(self, out_rows, out_cols, data_path = "data/train/image", label_path = "data/train/label", test_path = "data/test", npy_path = "npy_data", img_type = "png"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    #convert training set to numpy array
    def create_train_data(self):
    
        i = 0
        print('-'*30)
        print('Creating training images...')
        print('-'*30)
        #search existing images
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        for imgname in imgs:
            #get the filename without path
            midname = imgname[imgname.rindex("/")+1:]
            img = load_img(self.data_path + "/" + midname,grayscale = True)
            label = load_img(self.label_path + "/" + midname,grayscale = True)
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                #print progress
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        #save training set data
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        print(imgdatas)
        #save training set label
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')
    
    #convert testing set to numpy array, very similar to create_train_data()
    def create_test_data(self):
    
        i = 0
        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        imgs = glob.glob(self.test_path+"/*."+self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            img = load_img(self.test_path + "/" + midname,grayscale = True)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    #load and return training set - (data,label)
    def load_train_data(self):
        print('-'*30)
        print('load train images...')
        print('-'*30)
        imgs_train = np.load(self.npy_path+"/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
        #convert to float
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        #normalization
        imgs_train /= 255
        imgs_mask_train /= 255
        #binarization
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train,imgs_mask_train
        
    #load and return testing set - data
    def load_test_data(self):
        print('-'*30)
        print('load test images...')
        print('-'*30)
        imgs_test = np.load(self.npy_path+"/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

if __name__ == "__main__":

    mydata = dataProcess(512,512)
    mydata.create_train_data()
    mydata.create_test_data()
