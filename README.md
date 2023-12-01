This is the README of the Airport maintenance and fault detection project, done during my 2-month internship at Shanghai Shengtong Information Technology Co., Ltd.

The goal of this project is to utlize machine learning techniques to perform image classification and feature identification tasks to enable the low cost and accuate falut detection 
on mission critical componments such as jet engines, fuel lines, etc.

The general development outcome should be that any inexperienced airport worker can simply take several pictures of the critical componment in need of inspection at various angles,
and then upload those images to the program, the software will then outline the areas which many need attention (if there is a fault) as persentage confidence, or report all clear otherwise.

Currently, the difficulties which needs to be resolved are:
  -Variaty in training datasets.
  -The worker may take photos at various extreme angles, which are not likely to be covered by training dataset.
  -General quality of the input image, real world conditions may very which can cause changes in factors such as contrast, exposure, etc

To achieve this objjective, I am desiging and implementing a spatial transformer module on top of a 11-layer CNN outfitted with attention mechanisms. 


Refrences:
https://arxiv.org/pdf/1412.7755.pdf
https://arxiv.org/abs/1506.02025
