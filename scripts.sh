seed=42
iters=3000
batch_size=2000
optimizer="rmsprop"

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
for instance in $(cat qap_instances.txt); do
    python main_gpu3.py --instance $instance --seed $seed --iters $iters --batch_size $batch_size --optimizer $optimizer
done



# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp==========" >> result_gurobi.txt
# for instance in $(cat qap_instances.txt); do
#     python test_baseline.py --instance $instance
# done