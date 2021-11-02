# Description

Visual saliency refers a part in a scene that captures our attention. Conventionally, many saliency detection techniques, including deep convolutional neural networks (CNNs) based approaches, try to predict where people look without considering temporal information. However, temporal ordering of the fixations includes important cues about which part of the image may be more prominent. This project experiments with time weighted saliency maps or saliency volumes given the eye-fixations, ground truth saliency maps and images.

# Setup

- Download the images, fixations and fixation maps from http://salicon.net/challenge-2017/ and place the folders in data/
- Download the pre-trained PNAS weights from https://iiitaphyd-my.sharepoint.com/personal/samyak_j_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsamyak%5Fj%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FPNASNet%2D5%5FLarge%2Epth&parent=%2Fpersonal%2Fsamyak%5Fj%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments&originalPath=aHR0cHM6Ly9paWl0YXBoeWQtbXkuc2hhcmVwb2ludC5jb20vOnU6L2cvcGVyc29uYWwvc2FteWFrX2pfcmVzZWFyY2hfaWlpdF9hY19pbi9FUnBzYzgyc2hGSk5odG4teGZScjY5QUJDSHRKTlVsU0hrU2M5OXNyQXJEdFFRP3J0aW1lPThld0dlN085MkVn and place it in the folder PNAS/
- Run "python generate_saliency_volumes --time_slices N" where N is the number of your choice (typically between 2 and 10)
- Run "python train_vol.py --time_slices N --model_val_path model.pt --loss_vol_coeff 0.1" to train the model
- Run "python test.py --time_slices N --model_val_path model.pt --val_img_dir path/to/test/images --results_dir path/to/results" to generate predictions
