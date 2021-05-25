


first we load simple dataset
then we created function to return complete mature model
then we trained our model with trained data
here our goal is to evaluate test data with new created and not trained model with the help of old trained model

we have several ways to do it. lets see one by one

1) Checkpoint clallback options
this option for to saving the model weights while model is training
then load those wights for newly created model

2) save weights manually
in this section we save model manually and load it for newly created model.

3) save entire model
this section has two way 

one is that: we use tensorflow function model save
this method save each property of model architecture, weights, configuration

two of them is that: hdf5 file which also save architecture model, while other properties syou should do it manually.   