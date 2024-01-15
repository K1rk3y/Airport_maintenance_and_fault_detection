This is the README of the Airport maintenance and fault detection project, done during my 2-month internship at Shanghai Shengtong Information Technology Co., Ltd.

The goal of this project is to utlize machine learning techniques to perform image classification and feature identification tasks to enable the low cost and accuate falut detection 
on mission critical componments such as jet engines, fuel lines, etc.

The general development outcome should be that any inexperienced airport worker can simply take several pictures of the critical componment in need of inspection at various angles,
and then upload those images to the program, the software will then outline the areas which many need attention (if there is a fault) as persentage confidence, or report all clear otherwise.

Currently, the difficulties which needs to be resolved are:
  -Variaty in training datasets.
  -The worker may take photos at various extreme angles, which are not likely to be covered by training dataset.
  -General quality of the input image, real world conditions may very which can cause changes in factors such as contrast, exposure, etc

To achieve this objective, I am desiging and implementing a spatial transformer module on top of a 11-layer CNN outfitted with attention mechanisms. 

UPDATE- Due to the limitation and difficulty of incorprating attention mechanism in CNN, which is likely to result in a noticible performance
uplift in this application due to the high number of miscellaneous elements present in the input image. Instead I am developing in parallel a Vision Transformer module with high number of self attention layers in addition to a spatial transformer. 

UPDATE- There are now two pathways I have devised for the desiggn of this project, the Detection Transformer DETR is the state of the art solution to image recogonition, it uses the addition of a CNN to preprocess the input feature map to extract high level features, and a transformer decoder to enable bipartite matching and individual class label prediction heads. This is one of the solution I can pursue. 

The other however, is much more original and poetntially complex, therefore the applicability could be in question. I propose a MoE approach to image feature classification, but instead of using homogenus locational experts and a traditional matrix based FFD as a gate mechanism to achieve the same effect of a DETR, I use a secondary CNN / ViT to act as the gate mechanism which identifies and extracts locations based on pre-specified class labels. Then, each label will have a corresponding expert layer, which will be fulfiled by a DETR that is trained to identify a set of very specific feature in the extracted location by the gate network, the training will be done in parallel to the main model. Thus, this MoE approach will be able to focus on far more miniscule and obscure details than the standard solutions and can in theory be trained far more efficiently with less data.  


Refrences:
https://arxiv.org/pdf/1412.7755.pdf
https://arxiv.org/abs/1506.02025
https://arxiv.org/pdf/1409.0473.pdf
https://arxiv.org/pdf/2010.11929.pdf
