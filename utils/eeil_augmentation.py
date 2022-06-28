import numpy as np
import time
def data_augmentation_e2e(img, lab):
    """
        Realize the data augmentation in End-to-End paper
        Parameters
        ----------
        img: the original images, size = (n, c, w, h)
        lab: the original labels, size = (n)
        Returns
        ----------
        img_aug: the original images, size = (n * 12, c, w, h)
        lab_aug: the original labels, size = (n * 12)
    """
    shape = np.shape(img)
    img_aug = np.zeros((shape[0], 12, shape[1], shape[2], shape[3]))
    img_aug[:, 0, :, :, :] = img
    lab_aug = np.zeros((shape[0], 12))

    for i in range(shape[0]):
        np.random.seed(int(time.time()) % 1000)

        im = img[i]

        # brightness
        brightness = (np.random.rand(1)-0.5)*2*63
        im_temp = im + brightness

        img_aug[i, 1] = im_temp

        # constrast
        constrast = (np.random.rand(1)-0.5)*2*0.8+1
        m0 = np.mean(im[0])
        m1 = np.mean(im[1])
        m2 = np.mean(im[2])
        im_temp = im
        im_temp[0] = (im_temp[0]-m0)*constrast + m0
        im_temp[1] = (im_temp[1]-m1)*constrast + m1
        im_temp[2] = (im_temp[2]-m2)*constrast + m2
        img_aug[i, 2] = im_temp

        # crop
        im_temp = img_aug[i, :3]
        for j in range(3):
            x_ = int(np.random.rand(1)*1000)%8
            y_ = int(np.random.rand(1)*1000)%8
            im_temp = np.zeros(shape=(shape[1], shape[2]+8, shape[3]+8))
            im_temp[:, 4:-4, 4:-4] = img_aug[i, j]
            img_aug[i, 3+j] = im_temp[:, x_:x_+shape[2], y_:y_+shape[3]]



        # mirror
        for j in range(6):
            im_temp = img_aug[i, j]
            img_aug[i, 6 + j] = im_temp[:,-1::-1,:]

        lab_aug[i, :] = lab[i]

    idx = np.where(img_aug>255)
    img_aug[idx] = 255
    idx = np.where(img_aug<0)
    img_aug[idx] = 0

    img_aug = np.reshape(img_aug, newshape=(shape[0]*12, shape[1], shape[2], shape[3]))
    img_aug = np.array(img_aug, dtype=np.uint8)
    lab_aug = np.reshape(lab_aug, newshape=(shape[0]*12))
    lab_aug = np.array(lab_aug, dtype=np.int32)
    return img_aug, lab_aug
