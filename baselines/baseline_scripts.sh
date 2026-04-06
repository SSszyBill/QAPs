# gcc /home/xjx/A-xjx/QAPs/baselines/tabou_qap2.c -o /home/xjx/A-xjx/QAPs/baselines/tabou_qap2
# n=100
# for i in {9..256}; do
#     /home/xjx/A-xjx/QAPs/baselines/tabou_qap2 "/home/xjx/A-xjx/QAPs/qaplibs/Synthesis/QAP$n/qap${n}_${i}.dat"
# done


# n=50
# for i in {1..256}; do
#     /home/xjx/A-xjx/QAPs/baselines/tabou_qap2 "/home/xjx/A-xjx/QAPs/qaplibs/Synthesis/QAP$n/qap${n}_${i}.dat"
# done



#  /home/xjx/A-xjx/QAPs/baselines/tabou_qap2 qaplibs/TaillardE/tai175e20.dat


#  # 不使用时间限制（默认）
# ./tabou_qap2 data_file.txt

# # 使用随机种子，无时间限制
# ./tabou_qap2 data_file.txt 12345

# # 使用随机种子和时间限制（例如60秒）
# ./tabou_qap2 data_file.txt 12345 60.0

# # 只使用时间限制，随机种子使用当前时间
# ./tabou_qap2 data_file.txt 0 60.0



# # 不使用时间限制（默认）
# ./BMA2 data_file.dat

# # 使用时间限制（例如60秒）
# ./BMA2 data_file.dat 60.0

# g++ /home/xjx/A-xjx/QAPs/baselines/BMA2.cpp -o /home/xjx/A-xjx/QAPs/baselines/BMA2 -O3
# /home/xjx/A-xjx/QAPs/baselines/BMA2 filename


# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# for instance in $(cat qap_instances.txt); do
#     /home/xjx/A-xjx/QAPs/baselines/tabou_qap2 qaplibs/$instance.dat 42 1800.0
# done


timestamp=$(date +"%Y-%m-%d %H:%M:%S")
for instance in $(cat qap_instances.txt); do
    /home/xjx/A-xjx/QAPs/baselines/BMA2 /home/xjx/A-xjx/QAPs/qaplibs/$instance.dat 1800.0
done