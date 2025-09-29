import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import os
# Hint mask

def create_mask(data, miss_rate):
    n, d = data.shape
    mask = np.random.binomial(1, 1 - miss_rate, size=(n, d))
    return mask

def sample_z(n, d):
    return np.random.uniform(0., 1., size=(n, d)).astype(np.float32)

def sample_hint(mask, hint_rate): #hint H = b * mask + 0.5 * (1 - b)
    b = np.random.binomial(1, hint_rate, mask.shape).astype(np.float32)
    H = b * mask + 0.5 * (1 - b) # hint matrix H
    return H.astype(np.float32), b.astype(np.float32)

# Model

def generator(input_dim, hidden_dim=128):
    inp = layers.Input(shape=(input_dim * 2,))
    x = layers.Dense(hidden_dim, activation='relu')(inp)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    out = layers.Dense(input_dim, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out, name='Generator')

def discriminator(input_dim, hidden_dim=128):
    inp = layers.Input(shape=(input_dim * 2,))
    x = layers.Dense(hidden_dim, activation='relu')(inp)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    out = layers.Dense(input_dim, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out, name='Discriminator')

class GAIN:
    def __init__(self, data_dim, hint_rate=0.9, alpha=10.0, gen_hidden=128, dis_hidden=128, lr=1e-3):
        self.d = data_dim
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.G = generator(self.d, gen_hidden)
        self.D = discriminator(self.d, dis_hidden)
        self.g_opt = optimizers.Adam(lr)
        self.d_opt = optimizers.Adam(lr)

        self.bce = losses.BinaryCrossentropy(from_logits=False)
        self.mse = losses.MeanSquaredError()
    
    def comp_G_input(self, x, m, z):
        x_hat = m * x + (1 - m) * z
        return np.concatenate([x_hat, m], axis=1).astype(np.float32)
    def comp_D_input(self, x_tilde, hint):
        return np.concatenate([x_tilde, hint], axis=1).astype(np.float32)
    
    def train(self, X, mask, batch_size=128, epochs=1000, verbose=1):
        n = X.shape[0]
        for epoch in range(1, epochs+1):
            idx = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                x_mb = X[batch_idx].astype(np.float32)
                m_mb = mask[batch_idx].astype(np.float32)
                z_mb = sample_z(len(batch_idx), self.d)

                x_mb_tf = tf.constant(x_mb, dtype=tf.float32)
                m_mb_tf = tf.constant(m_mb, dtype=tf.float32)
                #train discriminator 
                with tf.GradientTape() as tape_d:
                    g_input = self.comp_G_input(x_mb, m_mb, z_mb)
                    g_sample = self.G(g_input, training=True)

                    x_tilde = m_mb_tf * x_mb_tf + (1 - m_mb_tf) * g_sample

                    hint_mb, b_mb = sample_hint(m_mb, self.hint_rate)

                    d_input = self.comp_D_input(x_tilde.numpy(), hint_mb) 
                    d_prob = self.D(d_input, training=True)

                    mask_unrevealed_tf = tf.constant(1 - b_mb, dtype=tf.float32)
                    d_loss_elem = - (m_mb_tf * tf.math.log(d_prob + 1e-8) +
                                 (1 - m_mb_tf) * tf.math.log(1 - d_prob + 1e-8))
                    d_loss = tf.reduce_mean(mask_unrevealed_tf * d_loss_elem)

                grads_d = tape_d.gradient(d_loss, self.D.trainable_variables)
                self.d_opt.apply_gradients(zip(grads_d, self.D.trainable_variables))
                #train generator
                with tf.GradientTape() as tape_g:
                    g_input = self.comp_G_input(x_mb, m_mb, z_mb)
                    g_sample = self.G(g_input, training=True)

                    x_tilde = m_mb_tf * x_mb_tf + (1 - m_mb_tf) * g_sample
                    hint_mb, b_mb = sample_hint(m_mb, self.hint_rate)

                    d_input = self.comp_D_input(x_tilde.numpy(), hint_mb)
                    d_prob = self.D(d_input, training=False) 

                    mask_missing_unrevealed_tf = tf.constant((1 - m_mb) * (1 - b_mb), dtype=tf.float32)
                    adv_loss_elem = -tf.math.log(d_prob + 1e-8)
                    adv_loss = tf.reduce_mean(mask_missing_unrevealed_tf * adv_loss_elem)
                    recon_elem = tf.square(x_mb_tf - g_sample)
                    recon_loss = tf.reduce_sum(m_mb_tf * recon_elem) / (tf.reduce_sum(m_mb_tf) + 1e-8)
                    
                    g_loss = adv_loss + self.alpha * recon_loss
                
                grads_g = tape_g.gradient(g_loss, self.G.trainable_variables)
                self.g_opt.apply_gradients(zip(grads_g, self.G.trainable_variables))
            if verbose and epoch % (max(1, epochs // 10)) == 0:
                print(f"Epoch {epoch}/{epochs}  |  D_loss={d_loss.numpy():.4f}  G_loss={g_loss.numpy():.4f}  adv={adv_loss.numpy():.4f}  recon={recon_loss.numpy():.6f}")
    def impute(self, x_incomplete, mask):
        z = sample_z(x_incomplete.shape[0], self.d)
        g_input = self.comp_G_input(x_incomplete.astype(np.float32), mask.astype(np.float32), z)
        g_out = self.G(g_input, training=False).numpy()
        imputed = mask * x_incomplete + (1 - mask) * g_out
        return imputed
    def save_model(self, path):
        print(f"Đang lưu model vào thư mục: {path}")
        os.makedirs(path, exist_ok=True)
        generator_filepath = os.path.join(path, 'generator.keras')
        discriminator_filepath = os.path.join(path, 'discriminator.keras')
        self.G.save(generator_filepath)
        self.D.save(discriminator_filepath)
        print("Lưu model thành công!")

    def load_model(self, path):
        print(f"Đang tải model từ thư mục: {path}")
        generator_filepath = os.path.join(path, 'generator.keras')
        discriminator_filepath = os.path.join(path, 'discriminator.keras')
        if os.path.exists(generator_filepath):
            self.G = tf.keras.models.load_model(generator_filepath)
        else:
            raise FileNotFoundError(f"Không tìm thấy file model Generator tại: {generator_filepath}")
        if os.path.exists(discriminator_filepath):
            self.D = tf.keras.models.load_model(discriminator_filepath)
        else:
            print(f"Cảnh báo: Không tìm thấy file model Discriminator tại: {discriminator_filepath}")
        print("Tải model thành công!")


        