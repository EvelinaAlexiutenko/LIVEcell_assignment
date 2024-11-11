import tensorflow as tf
from keras.losses import binary_crossentropy
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    """combine DICE and BCE"""
    return 0.01 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

    # IoU Metric for a given threshold


def iou_thresholded(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Cast predictions to float32
    y_true_float = tf.cast(y_true, tf.float32)  # Ensure y_true is also float32

    intersection = tf.reduce_sum(y_true_float * y_pred)
    union = tf.reduce_sum(y_true_float + y_pred) - intersection 

    return (intersection + 1e-7) / (union + 1e-7)


# Mean IoU over thresholds
def mean_iou(y_true, y_pred):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    iou_scores = [
        iou_thresholded(y_true, y_pred, threshold) for threshold in thresholds
    ]
    return tf.reduce_mean(iou_scores)


# Average False Negative Rate (AFNR)
def afnr(y_true, y_pred):
    thresholds = [0.50 + i * 0.05 for i in range(10)]
    fn_rates = []
    for threshold in thresholds:
        y_pred_thresh = tf.cast(y_pred > threshold, tf.float32)
        
        # Calculate false negatives
        false_negatives = tf.reduce_sum(tf.cast((tf.cast(y_true, tf.float32) == 1) & (y_pred_thresh == 0), tf.float32))
        
        # Calculate total positives (ground truth instances)
        positives = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
        
        # Calculate FNR for this threshold
        fn_rate = false_negatives / (positives + 1e-7)
        fn_rates.append(fn_rate)
    
    # Return the mean FNR across all thresholds to get AFNR
    return tf.reduce_mean(fn_rates)

# Average Precision (AP) Metric
def ap_metric(y_true, y_pred):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    ap_scores = []
    for threshold in thresholds:
        y_pred_thresh = tf.cast(y_pred > threshold, tf.float32)
        true_positives = tf.reduce_sum(
            tf.cast(
                (tf.cast(y_true, tf.float32) == 1) & (y_pred_thresh == 1), tf.float32
            )
        )
        false_positives = tf.reduce_sum(
            tf.cast(
                (tf.cast(y_true, tf.float32) == 0) & (y_pred_thresh == 1), tf.float32
            )
        )
        positives = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (positives + 1e-7)
        ap_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        ap_scores.append(ap_score)
    return tf.reduce_mean(ap_scores)
