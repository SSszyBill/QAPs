# seed=42
# iters=2000
# batch_size=1000

# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp==========" >> result.txt
# for instance in $(cat qap_instances.txt); do
#     python main_gpu.py --instance $instance --seed $seed --iters $iters --batch_size $batch_size
# done



timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "==========Running script at $timestamp==========" >> result_gurobi.txt
for instance in $(cat qap_instances.txt); do
    python test_baseline.py --instance $instance
done