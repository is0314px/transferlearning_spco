* /data_creation_sorce/: contains source codes for creating data

* /gibbs_dataset/: contains datasets aqcuired in multiple environments
    * /dataset_path_txt/: contains paths of datasets or learning results or mutual informations for learning
    * /realworld/: contains real world dataset, which are in aqcuired in Ritsumeikan university laboratories (without location name information)
    * /sigverse/: contains virtual world (SIGVerse) dataset
    * /sigverse_subject/: contains virtual world (SIGVerse) dataset, however multiple subjects teaches location names in the sentences

* /gibbs_source/: contains source codes for learning and prediction
    * Experiment_sigverse_general.sh: conducts learning and prediction and evaluation for general locations in each models
    * Experiment_sigverse_specific.sh: conducts learning and prediction and evaluation for home-specific locations in each models
    * gibbs_spco_transfer20+MI.py: learns parameters of the proposed model (spatial concept transfer knowledge model)

* /one_exsample_of_spco/: contains one exsample of spatial concept in a new environment of the proposed model. The model is taught home-specific names (Emma's-room, mother's-room, father's-room) in three bedroom, however it is not taught in onether locations (kitchen, toilet, etc).