# Author
My name is Woosuk (Alex) Byun and I recently received a master of computer science (machine learning) from Rochester Institute of Technology.

# Used Hardware: 1 NVIDIA A100-SXM4-80GB 

# Dataset
  The dataset is divided into the following: a training dataset, a test dataset and a validation dataset.
  The training dataset has 5,969 samples. The test dataset has 1,280 data samples. The validation dataset has 1,279 data samples.
 
# Training 
  **Hyperparameters**
  - Learning rate: `3e-5`
  - Epochs: `1`
  - Batch size: `16`

  The above are the hyperparameters I used so far. I will further tune these hyperparameters in future.

  All model parameters are loaded onto the gpu. 
  Current batch of 1-lead ecgs and their corresponding labels are loaded onto NVIDIA A100-SXM4-80GB from the ram.  
  Then, each 1-lead ecg in the current batch is converted into a matrix of size 140x512.      
  And, mean pooling is applied to each matrix of size 140x512 to produce each vector of size 512.  
  Then, each vector of size of 512 was converted into 4 logits.  
  And the average cross entropy loss (average of each cross entropy loss of each 1 lead ecg) was calculated. 
  All model parameters' partial derivatives are set to None in each iteration. Then, a subset of parameters's partial derivatives are calculated. 
  Then, the parameters that have corresponding partial derivatives will be updated. 
  Lastly, the average cross entropy loss for the current batch of 1-lead ecgs is added to total_loss. 
  Each the sum of each batch's average cross entropy loss will be divided by 374.
   
  **Model Architecture** 
  
     
   
  
  
  
   
  
      
