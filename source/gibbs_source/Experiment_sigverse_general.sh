#! /bin/sh

for n in 1LDK 2LDK 3LDK
do

#SpCo Transfer 20 (+MI) 0,1,2,4,8,16 env.
for e in 0 1 2 4 8 16
do
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary.txt spcotransfer20_${e} general #Check!! You can remove results.
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer20.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer20_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer20_${e}.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer20_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer20_${e} transfer_sigverse/name_dictionary.txt general ${n}
/usr/bin/python2.7 predict_position_and_evaluate_in_${n}_general.py gibbs_result/sigverse_result/spcotransfer20_${e} ${e} transfer_sigverse/name_dictionary.txt
done

#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary.txt spcotransfer20+MI_${e} general #Check!! You can remove results.
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer20+MI.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer20+MI_${e}.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary.txt general ${n}
/usr/bin/python2.7 predict_position_and_evaluate_in_${n}_general.py gibbs_result/sigverse_result/spcotransfer20+MI_${e} ${e} transfer_sigverse/name_dictionary.txt
done
done

#SpCo Transfer 19 (+MI) 16 env. 
for e in 16
do
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary.txt spcotransfer19_${e} general #Check!! You can remove results.
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer19.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer19_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer19_${e}.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer19_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer19_${e} transfer_sigverse/name_dictionary.txt general ${n}
/usr/bin/python2.7 predict_position_and_evaluate_in_${n}_general.py gibbs_result/sigverse_result/spcotransfer19_${e} ${e} transfer_sigverse/name_dictionary.txt
done

#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary.txt spcotransfer19+MI_${e} general #Check!! You can remove results.
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer19+MI.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer19+MI_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer19+MI_${e}.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer19+MI_${e} transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_${n}.txt gibbs_result/sigverse_result/spcotransfer19+MI_${e} transfer_sigverse/name_dictionary.txt general ${n}
/usr/bin/python2.7 predict_position_and_evaluate_in_${n}_general.py gibbs_result/sigverse_result/spcotransfer19+MI_${e} ${e} transfer_sigverse/name_dictionary.txt
done
done

#SpCoA (+MI) 16 env.
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary.txt spcoa general #Check!! You can remove results.
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spcoa.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num0_${n}.txt gibbs_result/sigverse_result/spcoa transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcoa.txt 0
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num0_${n}.txt gibbs_result/sigverse_result/spcoa transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num0_${n}.txt gibbs_result/sigverse_result/spcoa transfer_sigverse/name_dictionary.txt general ${n}
/usr/bin/python2.7 predict_position_and_evaluate_in_${n}_general.py gibbs_result/sigverse_result/spcoa 0 transfer_sigverse/name_dictionary.txt
done

#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary.txt spcoa+MI general #Check!! You can remove results.
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spcoa+MI.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num0_${n}.txt gibbs_result/sigverse_result/spcoa+MI transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcoa+MI.txt 0
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num0_${n}.txt gibbs_result/sigverse_result/spcoa+MI transfer_sigverse/name_dictionary.txt general
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num0_${n}.txt gibbs_result/sigverse_result/spcoa+MI transfer_sigverse/name_dictionary.txt general ${n}
/usr/bin/python2.7 predict_position_and_evaluate_in_${n}_general.py gibbs_result/sigverse_result/spcoa+MI 0 transfer_sigverse/name_dictionary.txt
done
done

#Output result
/usr/bin/python2.7 make_bar_graph_for_name_accuracy.py sigverse_result
/usr/bin/python2.7 make_bar_graph_for_position_accuracy.py sigverse_result
/usr/bin/python2.7 output_name_accuracy_at_each_general_place.py sigverse_result transfer_sigverse/name_dictionary.txt
/usr/bin/python2.7 output_position_accuracy_at_each_general_place.py sigverse_result transfer_sigverse/name_dictionary.txt