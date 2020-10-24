import numpy as np
import nibabel as nib
import os

def extract_segment(I, S_size, coordinate=None, subsample_factor=(1, 1, 1), default_value=0):
    assert len(I.shape) == len(S_size) + 1, "Number of image dimensions must be number of requested segment sizes + 1."
    coordinate = coordinate or [s // 2 for s in I.shape[:-1]]
    S_size = S_size + (I.shape[-1],)
    S = default_value * np.ones(S_size, dtype=np.float32)
    idx_I = [slice(None)] * I.ndim
    idx_S = [slice(None)] * S.ndim
    for i, (d_I, d_S, c, s_f) in enumerate(zip(I.shape[:-1], S_size[:-1], coordinate, subsample_factor)):
        n_left_I = c
        n_right_I = d_I - c - 1
        n_left_S = d_S // 2
        n_right_S = d_S // 2
        if d_S % 2 == 0:
            n_right_S -= 1

        if n_left_I < n_left_S * s_f:
            n = n_left_I // s_f
            start_S = d_S // 2 - n
            start_I = c - n * s_f

        else:
            start_S = 0
            start_I = c - n_left_S * s_f

        if n_right_I < n_right_S * s_f:
            n = n_right_I // s_f
            end_S = d_S // 2 + n
            end_I = c + n * s_f

        else:
            end_S = d_S - 1
            end_I = c + n_right_S * s_f

        idx_I[i] = slice(start_I, end_I + 1, s_f)
        idx_S[i] = slice(start_S, end_S + 1)

    S[tuple(idx_S)] = I[tuple(idx_I)]
    return S


if __name__ == "__main__":
    nii_path = os.path.join("D:\Project\Data\calibration","ISLES_2017\Data","case_0","MR_rCBV_2mm.nii.gz")
    nii = nib.load(nii_path)
    array = nii.get_data()[..., None]
    print("Array shape before cropping: ", array.shape)
    print("min: {}, max: {}".format(array.min(), array.max()))
    array_ = extract_segment(array, (136, 136, 82))
    print("Array shape after cropping: ", array_.shape)
    print("min: {}, max: {}".format(array_.min(),array_.max()))
