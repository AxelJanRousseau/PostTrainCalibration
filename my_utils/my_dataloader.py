import os
import numpy as np
import math
import nibabel as nib
from my_utils.cropping import extract_segment
from torch.utils.data import Dataset

BratsFolds = {0: ['case_5', 'case_7', 'case_8', 'case_12', 'case_15', 'case_22', 'case_45', 'case_55', 'case_59',
                  'case_63', 'case_64', 'case_73', 'case_74', 'case_76', 'case_81', 'case_90', 'case_92', 'case_97',
                  'case_101', 'case_103', 'case_106', 'case_110', 'case_111', 'case_124', 'case_126', 'case_136',
                  'case_150', 'case_154', 'case_158', 'case_159', 'case_171', 'case_179', 'case_181', 'case_188',
                  'case_191', 'case_194', 'case_200', 'case_201', 'case_205', 'case_207', 'case_208', 'case_213',
                  'case_218', 'case_222', 'case_225', 'case_227', 'case_236', 'case_239', 'case_241', 'case_252',
                  'case_267', 'case_270', 'case_271', 'case_278', 'case_279', 'case_280', 'case_284'],
              1: ['case_16', 'case_18', 'case_20', 'case_21', 'case_27', 'case_29', 'case_33', 'case_37', 'case_44',
                  'case_46', 'case_54', 'case_56', 'case_60', 'case_71', 'case_75', 'case_83', 'case_89', 'case_109',
                  'case_116', 'case_118', 'case_129', 'case_134', 'case_135', 'case_137', 'case_139', 'case_145',
                  'case_146', 'case_152', 'case_153', 'case_156', 'case_161', 'case_166', 'case_167', 'case_168',
                  'case_184', 'case_190', 'case_199', 'case_210', 'case_214', 'case_215', 'case_220', 'case_223',
                  'case_224', 'case_228', 'case_230', 'case_233', 'case_234', 'case_235', 'case_245', 'case_246',
                  'case_250', 'case_257', 'case_266', 'case_268', 'case_272', 'case_274', 'case_277'],
              2: ['case_2', 'case_3', 'case_4', 'case_10', 'case_13', 'case_14', 'case_19', 'case_24', 'case_26',
                  'case_30', 'case_40', 'case_41', 'case_43', 'case_50', 'case_51', 'case_52', 'case_58', 'case_61',
                  'case_62', 'case_66', 'case_67', 'case_77', 'case_80', 'case_86', 'case_96', 'case_104',
                  'case_107', 'case_108', 'case_122', 'case_123', 'case_125', 'case_130', 'case_144', 'case_155',
                  'case_157', 'case_160', 'case_173', 'case_175', 'case_176', 'case_182', 'case_187', 'case_189',
                  'case_198', 'case_206', 'case_212', 'case_216', 'case_217', 'case_219', 'case_221', 'case_229',
                  'case_238', 'case_247', 'case_254', 'case_260', 'case_262', 'case_264', 'case_265'],
              3: ['case_0', 'case_6', 'case_11', 'case_23', 'case_35', 'case_36', 'case_48', 'case_49', 'case_57',
                  'case_65', 'case_68', 'case_69', 'case_78', 'case_82', 'case_84', 'case_85', 'case_91', 'case_93',
                  'case_94', 'case_95', 'case_98', 'case_100', 'case_102', 'case_112', 'case_113', 'case_119',
                  'case_121', 'case_131', 'case_138', 'case_140', 'case_141', 'case_142', 'case_143', 'case_148',
                  'case_149', 'case_162', 'case_169', 'case_170', 'case_178', 'case_180', 'case_196', 'case_204',
                  'case_209', 'case_226', 'case_231', 'case_237', 'case_240', 'case_248', 'case_249', 'case_255',
                  'case_256', 'case_261', 'case_269', 'case_276', 'case_281', 'case_282', 'case_283'],
              4: ['case_1', 'case_9', 'case_17', 'case_25', 'case_28', 'case_31', 'case_32', 'case_34', 'case_38',
                  'case_39', 'case_42', 'case_47', 'case_53', 'case_70', 'case_72', 'case_79', 'case_87', 'case_88',
                  'case_99', 'case_105', 'case_114', 'case_115', 'case_117', 'case_120', 'case_127', 'case_128',
                  'case_132', 'case_133', 'case_147', 'case_151', 'case_163', 'case_164', 'case_165', 'case_172',
                  'case_174', 'case_177', 'case_183', 'case_185', 'case_186', 'case_192', 'case_193', 'case_195',
                  'case_197', 'case_202', 'case_203', 'case_211', 'case_232', 'case_242', 'case_243', 'case_244',
                  'case_251', 'case_253', 'case_258', 'case_259', 'case_263', 'case_273', 'case_275']
              }

Isles18Folds = {
    0: ['case_2', 'case_7', 'case_8', 'case_13', 'case_16', 'case_22', 'case_30', 'case_33', 'case_51', 'case_56',
        'case_60', 'case_61', 'case_62', 'case_73', 'case_76', 'case_78', 'case_79', 'case_89', 'case_93'],
    1: ['case_3', 'case_6', 'case_18', 'case_24', 'case_26', 'case_27', 'case_42', 'case_43', 'case_45', 'case_48',
        'case_54', 'case_55', 'case_66', 'case_74', 'case_77', 'case_82', 'case_84', 'case_86', 'case_87'],
    2: ['case_0', 'case_4', 'case_5', 'case_11', 'case_15', 'case_17', 'case_23', 'case_28', 'case_34', 'case_35',
        'case_38', 'case_40', 'case_41', 'case_50', 'case_53', 'case_59', 'case_63', 'case_81', 'case_85'],
    3: ['case_1', 'case_10', 'case_14', 'case_19', 'case_20', 'case_29', 'case_31', 'case_32', 'case_49', 'case_52',
        'case_57', 'case_68', 'case_69', 'case_71', 'case_75', 'case_80', 'case_88', 'case_91', 'case_92'],
    4: ['case_9', 'case_12', 'case_21', 'case_25', 'case_36', 'case_37', 'case_39', 'case_44', 'case_46', 'case_47',
        'case_58', 'case_64', 'case_65', 'case_67', 'case_70', 'case_72', 'case_83', 'case_90']}

class BratsDataset(Dataset):
    def __init__(self, data_path, fold, train=True, mask_output=True, mask_type='sigmoid', transforms=None):
        super().__init__()

        if train:
            self.cases = [case for k in BratsFolds if k != fold for case in BratsFolds[k]]
        else:
            self.cases = [case for case in BratsFolds[fold]]
        self.data_path = data_path
        self.transforms = transforms
        self.mask_output = mask_output

        self.shifts = [420.884688397679, 568.7868683246469, 639.4882077323609, 629.919352934067]
        self.scales = [1320.6450427506038, 1160.6822019612432, 1181.144425870453, 1363.6117673325714]
        self.mask_type = mask_type

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_path = os.path.join(self.data_path, case)

        raw_flair = nib.load(os.path.join(case_path, 'FLAIR_2mm.nii.gz')).get_fdata()
        flair = (raw_flair - self.shifts[0]) / self.scales[0]

        t1 = (nib.load(os.path.join(case_path, 'T1_2mm.nii.gz')).get_fdata() - self.shifts[1]) / self.scales[1]
        t1_ce = (nib.load(os.path.join(case_path, 'T1_CE_2mm.nii.gz')).get_fdata() - self.shifts[2]) / self.scales[2]
        t2 = (nib.load(os.path.join(case_path, 'T2_2mm.nii.gz')).get_fdata() - self.shifts[3]) / self.scales[3]
        gt = (nib.load(os.path.join(case_path, 'GT_W_2mm.nii.gz')).get_fdata() )

        input1 = np.stack([flair, t1, t1_ce, t2], axis=-1)
        # input1 = np.pad(input1, [[21, 21], [21, 21], [15, 15], [0, 0]], 'constant')
        # gt = np.pad(gt, [[8, 8], [8, 8], [2, 2]], 'constant')
        input1 = extract_segment(input1, (162, 162, 108))
        gt = extract_segment(gt[...,None], (136, 136, 82))
        if self.mask_output:
            # input2 = np.pad(raw_flair > 0, [[8, 8], [8, 8], [2, 2]], 'constant').astype('float32')
            input2 = extract_segment(raw_flair[...,None] > 0, ( 136, 136, 82)).astype('float32')
            if self.mask_type == 'linear':
                input2 = np.log(input2 + 1e-45)  # input2 - 150 # of np.log(input2 + 1e-45)
        else:
            input2 = None

        return input1, input2, gt


class Isles18Dataset(Dataset):
    def __init__(self, data_path, fold, train=True, mask_output=True, mask_type='sigmoid', transforms=None):
        super().__init__()

        if train:
            self.cases = [case for k in Isles18Folds if k != fold for case in Isles18Folds[k]]
        else:
            self.cases = [case for case in Isles18Folds[fold]]
        self.data_path = data_path
        self.transforms = transforms
        self.mask_output = mask_output

        self.shifts = [145.4089064728558, 2.8753392425245097, 5.538641988928408, 179.03296070754575, 25.937894045251678]
        self.scales = [186.9740130523932, 4.634348718250107, 5.2019667021425295, 293.8784533196274, 42.502019634374086]
        self.mask_type = mask_type

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_path = os.path.join(self.data_path, case)

        CT_2mm_raw = nib.load(os.path.join(case_path, 'CT_2mm.nii.gz')).get_fdata()

        CT_2mm = (CT_2mm_raw - self.shifts[0]) / self.scales[0]
        CT_Tmax_2mm = (nib.load(os.path.join(case_path, 'CT_Tmax_2mm.nii.gz')).get_fdata() - self.shifts[1]) / \
                      self.scales[1]
        CT_MTT_2mm = (nib.load(os.path.join(case_path, 'CT_MTT_2mm.nii.gz')).get_fdata() - self.shifts[2]) / \
                     self.scales[2]
        CT_CBF_2mm = (nib.load(os.path.join(case_path, 'CT_CBF_2mm.nii.gz')).get_fdata() - self.shifts[3]) / \
                     self.scales[3]
        CT_CBV_2mm = (nib.load(os.path.join(case_path, 'CT_CBV_2mm.nii.gz')).get_fdata() - self.shifts[4]) / \
                     self.scales[4]
        gt = (nib.load(os.path.join(case_path, 'OT_2mm.nii.gz')).get_fdata())

        input1 = np.stack([CT_2mm, CT_Tmax_2mm, CT_MTT_2mm, CT_CBF_2mm, CT_CBV_2mm], axis=-1)
        input1 = extract_segment(input1, (162, 162, 108))
        gt = extract_segment(gt[...,None], (136, 136, 82))
        if self.mask_output:
            input2 = CT_2mm_raw > -23
            input2 = extract_segment(input2[...,None], (136, 136, 82)).astype('float32')
            if self.mask_type == 'linear':
                input2 = np.log(input2 + 1e-45)  # input2 - 150 # of np.log(input2 + 1e-45)
        else:
            input2 = None

        return input1, input2, gt


class Isles17Dataset(Dataset):
    def __init__(self, data_path, fold, train=True, mask_output=True, mask_type='sigmoid', transforms=None):
        super().__init__()

        if train:
            self.cases = [case for k in Isles17Folds if k != fold for case in Isles17Folds[k]]
        else:
            self.cases = [case for case in Isles17Folds[fold]]
        self.data_path = data_path
        self.transforms = transforms
        self.mask_output = mask_output

        self.shifts = [918.5765819726356, 6.306370011286291, 5.1032563033778, 28.844982342424302, 18.97638790517183,
                       1.8225750531150535]
        self.scales = [655.0562518031873, 10.26716165312032, 4.1382274944829645, 16.030510609259178, 28.87138245597804,
                       3.5792610614453815]
        self.mask_type = mask_type

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_path = os.path.join(self.data_path, case)

        MR_ADC_2mm_raw = nib.load(os.path.join(case_path, 'MR_ADC_2mm.nii.gz')).get_fdata()

        MR_ADC_2mm = (MR_ADC_2mm_raw - self.shifts[0]) / self.scales[0]
        MR_Tmax_2mm = (nib.load(os.path.join(case_path, 'MR_Tmax_2mm.nii.gz')).get_fdata() - self.shifts[1]) / \
                      self.scales[1]
        MR_MTT_2mm = (nib.load(os.path.join(case_path, 'MR_MTT_2mm.nii.gz')).get_fdata() - self.shifts[2]) / \
                     self.scales[2]
        MR_TTP_2mm = (nib.load(os.path.join(case_path, 'MR_TTP_2mm.nii.gz')).get_fdata() - self.shifts[3]) / \
                     self.scales[3]
        MR_rCBF_2mm = (nib.load(os.path.join(case_path, 'MR_rCBF_2mm.nii.gz')).get_fdata() - self.shifts[4]) / \
                      self.scales[4]
        MR_rCBV_2mm = (nib.load(os.path.join(case_path, 'MR_rCBV_2mm.nii.gz')).get_fdata() - self.shifts[5]) / \
                      self.scales[5]
        gt = (nib.load(os.path.join(case_path, 'OT_2mm.nii.gz')).get_fdata() )

        input1 = np.stack([MR_ADC_2mm, MR_Tmax_2mm, MR_MTT_2mm, MR_TTP_2mm, MR_rCBF_2mm, MR_rCBV_2mm], axis=-1)
        # input1 = pad_input(input1, (162, 162, 108, 6))
        # gt = pad_input(gt, (136, 136, 82))
        input1 = extract_segment(input1, (162, 162, 108))
        gt = extract_segment(gt[...,None], (136, 136, 82))
        if self.mask_output:
            input2 = MR_ADC_2mm_raw > 0
            input2 = extract_segment(input2[...,None], (136, 136, 82)).astype('float32')
            if self.mask_type == 'linear':
                input2 = np.log(input2 + 1e-45)  # input2 - 150 # of np.log(input2 + 1e-45)
        else:
            input2 = None

        return input1, input2, gt


class WMH17Dataset(Dataset):
    def __init__(self, data_path, fold, train=True, mask_output=True, mask_type='sigmoid', transforms=None):
        super().__init__()

        if train:
            self.cases = [case for k in WMH_2017Folds if k != fold for case in WMH_2017Folds[k]]
        else:
            self.cases = [case for case in WMH_2017Folds[fold]]
        self.data_path = data_path
        self.transforms = transforms
        self.mask_output = mask_output

        self.shifts = [130, 262]
        self.scales = [226, 490]
        self.mask_type = mask_type

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_path = os.path.join(self.data_path, case)

        T1_2mm_raw = nib.load(os.path.join(case_path, 'T1_2mm.nii.gz')).get_fdata()

        T1_2mm = (T1_2mm_raw - self.shifts[0]) / self.scales[0]
        FLAIR_2mm = (nib.load(os.path.join(case_path, 'FLAIR_2mm.nii.gz')).get_fdata() - self.shifts[1]) / \
                      self.scales[1]
        gt = (nib.load(os.path.join(case_path, 'wmh_2mm.nii.gz')).get_fdata() )

        input1 = np.stack([FLAIR_2mm, T1_2mm], axis=-1)
        input1 = pad_input(input1, (162, 162, 108, 2))
        gt = pad_input(gt, (136, 136, 82))
        if self.mask_output:
            input2 = T1_2mm_raw > 25
            input2 = pad_input(input2, (136, 136, 82)).astype('float32')
            if self.mask_type == 'linear':
                input2 = np.log(input2 + 1e-45)  # input2 - 150 # of np.log(input2 + 1e-45)
        else:
            input2 = None

        return input1, input2[..., None], gt


def pad_input(input, target_shape):
    pads = []
    in_shape = input.shape
    for i in range(len(target_shape)):
        pad = (target_shape[i] - in_shape[i]) / 2

        if pad <0:
            pad = pad *-1
            input = crop_input(input,[math.floor(pad), math.ceil(pad)])
            pad = 0
        pads.append([math.floor(pad), math.ceil(pad)])
    return np.pad(input, pads, 'constant')

def crop_input(input,pad):
    ## enkel voor WMH_17
    return input[:,:,pad[0]:-pad[1]]


def my_collate_fn(batch):
    in1, in2, gt = zip(*batch)
    in1 = np.stack(in1)
    in2 = np.stack(in2)
    gt = np.stack(gt)
    return in1, in2, gt


def my_collate_fn_for_tf_iterator(batch):
    in1, in2, gt = my_collate_fn(batch)
    return [in1, in2], gt
