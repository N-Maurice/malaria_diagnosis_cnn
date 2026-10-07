

Literature Review


Convolutional neural networks (CNNs) are well-suited to malaria diagnosis from blood smear images because they learn spatial features directly from pixel data, avoiding the hand-engineered rules earlier systems relied on, which struggled with the subtle variation in parasite shape and stain intensity. Rajaraman et al. (2018) compared a custom CNN against transfer-learning baselines on the same NIH/NLM dataset used in this project, finding both approaches exceeded 89 percent sensitivity and specificity, with transfer learning performing strongly when fine-tuned rather than used as a static feature extractor. This motivated the fine-tuning approach taken below.

EfficientNetB0 (Tan & Le, 2019) is the model chosen here due to its ability to optimally balance the depth, width, and resolution components of the compound scaling. EfficientNetB0 performs extremely well while minimizing problems of overfitting that can occur when deeper networks are employed on datasets containing visually similar cell crops.

Methodology: Transfer Learning Model 1 (EfficientNetB0) 

EfficientNetB0, which has been pretrained with ImageNet, used the plan of substituting the classification head with global average pooling, dropout, and a final sigmoid output layer suitable for binary classification. All the images in the dataset were artificially modified to have a size of 224 square pixels and were made ready for processing by EfficientNet's regular method. The dataset was divided into training, validation, and testing datasets, with a ratio of 70/15/15 and fixes seed used amongst the three models in the Group. Data augmentation techniques such as flipping and punching were utilized for the training procedure only.

Seven experiments isolated one variable each: frozen vs. fine-tuned layers, fine-tuning depth (20 vs. 40 layers), extended training with early stopping, optimizer (Adam vs. SGD), dropout rate, and batch size. The baseline (frozen) model reached 93.2 percent validation sensitivity; progressively deeper fine-tuning improved this, with the best configuration unfreezing the final 40 layers, trained 15 epochs with early stopping selected as the final model. 

Results

On the held-out test set, the final model reached 95.4 percent sensitivity and 96.8 percent specificity, exceeding the target of 95 percent and 90 percent respectively, with 96 percent overall accuracy and an AUC of 0.9925. Of 4,134 test images, 65 were false positives and 97 false negatives, meaning errors leaned slightly toward missed infections, the clinically costlier mistake. 

The training and validation curves showed that they were closely associated throughout all our seven experiments, with no deviation indicating high generalization. Grad-CAM made it possible for correctly classified infected cells to be identified in a heat map concentrated in various intra-cellular regions, whereas the cells that were not infected show diffusion and edge-based activation. The case examined of misclassification showed that the system put emphasis on a possible staining artifact as opposed to the original parasite mark. The overall analysis of all cases of misclassification pointed that most often false negatives have weak stains whereas false positives contain dark patches similar to the parasite marks without being related to real infection.
