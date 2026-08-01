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
  
  Current batch of 1-lead ecgs and their corresponding labels were loaded onto the gpu from the ram.
  Then, each 1-lead ecg is converted into a matrix of size 140x512. And, mean pooling was applied to each matrix of size 140x512 
  which represents each 1-lead ecg and now each 1-lead was represented by a vector of size 512. Then, each vector of size of 512 was converted 
  to 4 logits. And the average cross entropy (average of each cross entropy loss of each 1 lead ecg) loss was calculated. 
  Initially, all parameters' partial derivatives are set to None. Then, a subset of parameter's partial derivatives are calculated. 
  Then, the parameters that have corresponding partial derivatives will be updated. And the average cross entropy loss for the current batch
  of 1-lead ecgs is added to total_loss. There will in total 374 iterations of the aforementioned process.
  

  **Model Architecture** 
  
     
   
  
  
  
   
  
      
