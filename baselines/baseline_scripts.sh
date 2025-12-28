gcc /home/xjx/A-xjx/QAPs/baselines/tabou_qap2.c -o /home/xjx/A-xjx/QAPs/baselines/tabou_qap2
n=100
for i in {9..256}; do
    /home/xjx/A-xjx/QAPs/baselines/tabou_qap2 "/home/xjx/A-xjx/QAPs/qaplibs/Synthesis/QAP$n/qap${n}_${i}.dat"
done


n=50
for i in {1..256}; do
    /home/xjx/A-xjx/QAPs/baselines/tabou_qap2 "/home/xjx/A-xjx/QAPs/qaplibs/Synthesis/QAP$n/qap${n}_${i}.dat"
done