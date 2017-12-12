#!/bin/bash


#Flags
while getopts v:t:decay: aflag; do
    case $aflag in
        v) verbosity=$OPTARG
        
        ;;
        t) tone=$OPTARG
        
        ;;
        d) decay=$OPTARG
        ;;
    esac
done
echo "${decay}"
echo "The verbosity is ..${verbosity}.. and the tone is ..${tone}.."

#export JOB_NAME="Keras_imbd_wiki$(date +%Y%m%d_%H%M%S)"
#export JOB_DIR=gs://$BUCKET/$JOB_NAME 
#python /home/gabriel.barros/condor_test/my_cifar10.py

#python /home/gabriel.barros/condor_test/experiments/classifier_DCGAN_review.py

   

#python /home/gabriel.barros/condor_test/experiments/classifier_review_argparse.py 

#python classifier_argparse_teste.py --real_labelSmooth=0.1 --outputDir='outputdir_train_classifier_decay_0d010_label_0d10' --decay=0.1
echo "python classifier_argparse_teste.py --real_labelSmooth=0.1 --outputDir='outputdir_train_classifier_decay_0d010_label_0d10' --decay=${decay}"

#python /home/gabriel.barros/condor_test/experiments/cuda_parallel.py 
#python /home/gabriel.barros/condor_test/experiments/nvidia-smi.py 

