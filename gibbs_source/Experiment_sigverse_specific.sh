#! /bin/sh

#SpCo Transfer 19 16 env.
for e in 16
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary_specific.txt spcotransfer19_${e} specific #Check!! You can remove results.
do
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer19.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer19_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer19_${e}_specific.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer19_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer19_${e} transfer_sigverse/name_dictionary_specific.txt specific 3LDK
/usr/bin/python2.7 predict_position_and_evaluate_in_3LDK_specific.py gibbs_result/sigverse_result/spcotransfer19_${e} ${e} transfer_sigverse/name_dictionary_specific.txt
done
done

#SpCo Transfer 19 +MI 16 env.
for e in 16
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary_specific.txt spcotransfer19+MI_${e} specific #Check!! You can remove results.
do
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer19+MI.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer19+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer19+MI_${e}_specific.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer19+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 evaluate_name_prediction_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer19+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific 3LDK
/usr/bin/python2.7 predict_position_and_evaluate_in_3LDK_specific.py gibbs_result/sigverse_result/spcotransfer19+MI_${e} ${e} transfer_sigverse/name_dictionary_specific.txt
done
done

#SpCo Transfer 20 16 env.
for e in 16
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary_specific.txt spcotransfer20_${e} specific #Check!! You can remove results.
do
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer20.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer20_${e}_specific.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20_${e} transfer_sigverse/name_dictionary_specific.txt specific 3LDK
/usr/bin/python2.7 predict_position_and_evaluate_in_3LDK_specific.py gibbs_result/sigverse_result/spcotransfer20_${e} ${e} transfer_sigverse/name_dictionary_specific.txt
done
done

#SpCo Transfer 20 +MI 16 env.
for e in 16
#/usr/bin/python2.7 remove_result.py transfer_sigverse/name_dictionary_specific.txt spcotransfer20+MI_${e} specific #Check!! You can remove results.
do
for i in `seq 20`
do
/usr/bin/python2.7 gibbs_spco_transfer20+MI.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 calculate_mutual_information_for_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/mutual_path_spcotransfer20+MI_${e}_specific.txt ${e}
/usr/bin/python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific
/usr/bin/python2.7 evaluate_name_prediction.py ../gibbs_dataset/dataset_path_txt/transfer_sigverse/path_num${e}_3LDK.txt gibbs_result/sigverse_result/spcotransfer20+MI_${e} transfer_sigverse/name_dictionary_specific.txt specific 3LDK
/usr/bin/python2.7 predict_position_and_evaluate_in_3LDK_specific.py gibbs_result/sigverse_result/spcotransfer20+MI_${e} ${e} transfer_sigverse/name_dictionary_specific.txt
done
done

#Output result
/usr/bin/python2.7 make_line_graph_for_name_accuracy.py sigverse_result
/usr/bin/python2.7 make_line_graph_for_position_accuracy.py sigverse_result
/usr/bin/python2.7 output_name_accuracy_at_each_specific_place.py sigverse_result transfer_sigverse/name_dictionary_specific.txt
/usr/bin/python2.7 output_position_accuracy_prediction_at_each_specific_place.py sigverse_result transfer_sigverse/name_dictionary_specific.txt