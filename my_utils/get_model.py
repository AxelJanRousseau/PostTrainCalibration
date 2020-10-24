from my_utils import unet_generalized

def get_Brats_model(output_type, dropout_during_inference=False, dropout_common_pathway=[0], dropout_setting=None):
    dropout_decoders = False
    dropout_center = False
    if dropout_setting == "center":
        dropout_center = True
    if dropout_setting == "decoder":
        dropout_decoders = True
    model = unet_generalized.create_unet_like_model(
        number_input_features=4,
        subsample_factors_per_pathway=[
            (1, 1, 1),
            (3, 3, 3),
            (9, 9, 9),
            (27, 27, 27)
        ],
        kernel_sizes_per_pathway=[
            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
            [[(3, 3, 3), (3, 3, 3)], []]
        ],
        number_features_per_pathway=[
            [[20, 40], [40, 20]],
            [[40, 80], [80, 40]],
            [[80, 160], [160, 80]],
            [[160, 160], []]
        ],
        output_size=[136, 136, 82],
        padding='same',
        upsampling='linear',
        activation_final_layer=output_type,  # activation_final_layer='sigmoid', activation_final_layer='linear'
        mask_output=True,
        dropout_during_inference=dropout_during_inference,
        dropout_common_pathway=dropout_common_pathway,
        dropout_decoders=dropout_decoders,
        dropout_center=dropout_center,
        l2_reg=1e-5
    )

    return model

def get_Isles_18_model(output_type, dropout_during_inference=False, dropout_common_pathway=[0], dropout_setting=None):
    dropout_decoders = False
    dropout_center = False
    if dropout_setting == "center":
        dropout_center = True
    if dropout_setting == "decoder":
        dropout_decoders = True
    model = unet_generalized.create_unet_like_model(
        number_input_features=5,
        subsample_factors_per_pathway=[
            (1, 1, 1),
            (3, 3, 3),
            (9, 9, 9),
            (27, 27, 27)
        ],
        kernel_sizes_per_pathway=[
            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
            [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
            [[(3, 3, 3), (3, 3, 3)], []]
        ],
        number_features_per_pathway=[
            [[10, 20], [20, 10]],
            [[20, 40], [40, 20]],
            [[40, 80], [80, 40]],
            [[80, 80], []]
        ],
        output_size=[136, 136, 82],
        padding='same',
        pooling='avg',
        upsampling='linear',
        activation_final_layer=output_type,
        instance_normalization=False,
        mask_output=True,
        dropout_during_inference=dropout_during_inference,
        dropout_common_pathway=dropout_common_pathway,
        dropout_decoders=dropout_decoders,
        dropout_center=dropout_center,
        l2_reg=0
    )
    return model
