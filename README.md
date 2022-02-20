# Wafer map pattern classification using Stacking ensemble

Wafer map defect pattern classification using Stacking ensemble

Proposed by H.Kang and S.Kang

* Hyungu Kang, Seokho Kang* (2021), "A stacking ensemble classifier with handcrafted and convolutional features for wafer map pattern classification", Computers in Industry 129: 103450 (https://www.sciencedirect.com/science/article/pii/S0166361521000579?via%3Dihub)

## Methodology

Stacking Ensemble![](https://github.com/DMkelllog/WMPC_Stacking/blob/main/Stacking%20flow.PNG?raw=true)

* Input:    probability outputs from MFE and CNN
* Output: predicted class
* Model:  Stacking ensemble (FNN as the meta-classifer)

## Data

* WM811K
  * 811457 wafer maps collected from 46393 lots in real-world fabrication

  * 172950 wafers were labeled by domain experts.

  * 9 defect classes (Center, Donut, Edge-ring, Edge-local, Local, Random, Near-full, Scratch, None)

    

  * provided by MIR Lab (http://mirlab.org/dataset/public/)

  * .pkl file downloaded from Kaggle dataset (https://www.kaggle.com/qingyi/wm811k-wafer-map)

  * directory: /data/LSWMD.pkl

## Dependencies

* Python
* Pandas
* Tensorflow 2.1
* Scikit-learn
* Scikit-image

## References

* WM-811K(LSWMD). National Taiwan University Department of Computer Science Multimedia Information Retrieval LAB http://mirlab.org/dataSet/public/
* Kang, H., & Kang, S. (2021). A stacking ensemble classifier with handcrafted and convolutional features for wafer map pattern classification. Computers in Industry, 129, 103450.
* Nakazawa, T., & Kulkarni, D. V. (2018). Wafer map defect pattern classification and image retrieval using convolutional neural network. IEEE Transactions on Semiconductor Manufacturing, 31(2), 309-314.
* Shim, J., Kang, S., & Cho, S. (2020). Active learning of convolutional neural network for cost-effective wafer map pattern classification. IEEE Transactions on Semiconductor Manufacturing, 33(2), 258-266.
* Kang, S. (2020). Rotation-Invariant Wafer Map Pattern Classification With Convolutional Neural Networks. IEEE Access, 8, 170650-170658.
* Wu, M. J., Jang, J. S. R., & Chen, J. L. (2014). Wafer map failure pattern recognition and similarity ranking for large-scale data sets. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1-12.
* Fan, M., Wang, Q., & van der Waal, B. (2016, October). Wafer defect patterns recognition based on OPTICS and multi-label classification. In 2016 IEEE Advanced Information Management, Communicates, Electronic and Automation Control Conference (IMCEC) (pp. 912-915). IEEE.
* Saqlain, M., Jargalsaikhan, B., & Lee, J. Y. (2019). A voting ensemble classifier for wafer map defect patterns identification in semiconductor manufacturing. IEEE Transactions on Semiconductor Manufacturing, 32(2), 171-182.

