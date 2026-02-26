import numpy as np

def compute_modal_contribution(shap_values,
                               img_dim,
                               vid_dim,
                               aud_dim):

    image_score = np.mean(np.abs(shap_values[:, :img_dim]))
    video_score = np.mean(np.abs(shap_values[:, img_dim:img_dim+vid_dim]))
    audio_score = np.mean(np.abs(shap_values[:, -aud_dim:]))

    return {
        "image": float(image_score),
        "video": float(video_score),
        "audio": float(audio_score)
    }