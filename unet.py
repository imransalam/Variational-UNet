import tensorflow as tf

class VariationalUnet(tf.keras.Model):
    def __init__(self):
        super(VariationalUnet, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.mu1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.sigma1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.mu2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.sigma2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.mu3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.sigma3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.drop4 = tf.keras.layers.Dropout(0.5)
        self.mu4 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.sigma4 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv5_1 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5_2 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.drop5 = tf.keras.layers.Dropout(0.5)



        self.up6_1 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.up6_2 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up7_1 = tf.keras.layers.UpSampling2D(size = (2,2))
        self.up7_2 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up8_1 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.up8_2 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up9_1 = tf.keras.layers.UpSampling2D(size = (2,2))
        self.up9_2 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_3 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        
        self.conv10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
    

    def sample_from_mu_sigma(self, mu, sigma):
        std = tf.exp(0.5 * sigma)
        eps = tf.random_normal(shape=tf.shape(std))
        return mu + eps * std
        
    def call(self, x):

        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        mu1 = self.mu1(conv1_2)
        sigma1 = self.sigma1(conv1_2)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        mu2 = self.mu2(conv2_2)
        sigma2 = self.sigma2(conv2_2)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        mu3 = self.mu3(conv3_2)
        sigma3 = self.sigma3(conv3_2)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        drop4 = self.drop4(conv4_2)
        mu4 = self.mu4(drop4)
        sigma4 = self.sigma4(drop4)
        pool4 = self.pool4(drop4)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        drop5 = self.drop5(conv5_2)

        up6_1 = self.up6_1(drop5)
        up6_2 = self.up6_2(up6_1)


        merge6 = tf.keras.layers.concatenate([self.sample_from_mu_sigma(mu4, sigma4), up6_2], axis=3)
        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)

        up7_1 = self.up7_1(conv6_2)
        up7_2 = self.up7_2(up7_1)
        merge7 = tf.keras.layers.concatenate([self.sample_from_mu_sigma(mu3, sigma3), up7_2], axis=3)
        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)

        up8_1 = self.up8_1(conv7_2)
        up8_2 = self.up8_2(up8_1)
        merge8 = tf.keras.layers.concatenate([self.sample_from_mu_sigma(mu2, sigma2), up8_2], axis=3)
        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)

        up9_1 = self.up9_1(conv8_2)
        up9_2 = self.up9_2(up9_1)
        merge9 = tf.keras.layers.concatenate([self.sample_from_mu_sigma(mu1, sigma1), up9_2], axis=3)
        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)
        conv9_3 = self.conv9_3(conv9_2)

        conv10 = self.conv10(conv9_3)
        return conv10
