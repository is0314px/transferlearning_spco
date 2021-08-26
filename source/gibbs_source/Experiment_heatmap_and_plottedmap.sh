#! /bin/sh

#<< COMMENTOUT
for e in 16
do
for i in `seq 10`
do
/usr/bin/python2.7 gibbs_spco_transfer20+MI.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python3 make_heatmap_and_plottedmap.py gibbs_result/sigverse_result transfer_sigverse/name_dictionary_specific.txt ${i}
echo ${i}
done
done
#COMMENTOUT