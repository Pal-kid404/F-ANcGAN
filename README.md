# F-ANcGAN
Nanomaterial research is critical for advancements in energy, medicine, and materials science, but progress is often hampered by the scarcity of high-quality annotated datasets for nanoparticle topology analysis. To address this challenge, we present F-ANcGAN, a novel attention-enhanced cycle-consistent generative adversarial network (GAN) that generates realistic scanning electron microscopy (SEM) images directly from segmentation maps—even with limited training data. 
Read the full paper - http://export.arxiv.org/abs/2505.18106
# Dataset
The data used in this research is obtained from the GitHub repository( https://github.com/BAMresearch/automatic-sem-image-segmentation ). The dataset includes electron microscopy (EM) images of TiO_2 particle and their respective segmentation masks defining the nanoparticle boundaries.The repository is organized into sub-folders with raw scanning electron microscopy (SEM) and transmission scanning electron microscopy (TSEM) images and their respective manually labelled segmentation and classification masks. 
# Utilization
## Training F-ANcGAN
The model can be trained using either a user-supplied dataset or the TiO_2 dataset whose link is provided in this repository. Prior to training, users are advised to configure key hyperparameters and the early stopping patience parameters. Throughout the training process, the script records detailed logs and periodically saves model checkpoints for the  generator network (Gen.pth), segmentation network (Seg.pth), and discriminators (D1.pth and D2.pth) within the designated output_dir.
<pre>Example:
python train.py --seg_model UNET --train_set_dir  .../TiO_2 dataset/train  --lr 0.0001 --p_vanilla 0.2 --p_diff 0.2 --patience 500 --output_dir tmp/</pre>
## Testing the generator model
Evaluate the StyleUNET generator's performance, using the synthetic or real mask images using the following example code:
<pre>Example:
python test_generation_model.py --test_set_dir .../TiO2 dataset/test/ --gen_ckpt_dir .../SAGE-GAN_checkpoints/UNET_model/S1 dataset/Gen.pth --output_dir tmp/</pre>
## Testing the segmentation model
Evaluate the segmentation model using the S1 dataset or yours, specifying the segmentation model type (seg_model) and its checkpoint directory (seg_ckpt_dir).
<pre>Example:
python test_segmentation_model.py --seg_model UNET --test_set_dir .../TiO_2 dataset/test --seg_ckpt_dir .../SAGE-GAN_checkpoints/UNET_model/S1 dataset/Seg.pth --output_dir tmp/</pre>
# Useful Information
For any queries, contact anindyapal264@gmail.com / varunajith29@gmail.com .
