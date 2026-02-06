import pywt
import numpy as np


class WaveletTransformer:
    """
    Haar Wavelet Denoising Transformer
    """

    def __init__(self, wavelet="haar", level=1):
        self.wavelet = wavelet
        self.level = level

    def denoise_series(self, series):

        coeffs = pywt.wavedec(series, self.wavelet, level=self.level)

        # Zero out high-frequency coefficients
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]

        reconstructed = pywt.waverec(coeffs, self.wavelet)

        return reconstructed[:len(series)]

    def transform(self, df, columns):

        df_copy = df.copy()

        for col in columns:
            df_copy[col] = self.denoise_series(df_copy[col].values)

        return df_copy
