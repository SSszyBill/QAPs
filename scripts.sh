seed=42
iters=5000
batch_size=50
optimizer="rmsprop"

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
for instance in $(cat qap_instances.txt); do
    python main_gpu3.py --instance $instance --seed $seed --iters $iters --batch_size $batch_size --optimizer $optimizer
done



# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp==========" >> result_gurobi.txt
# for instance in $(cat qap_instances.txt); do
#     python test_baseline.py --instance $instance --time_limit 1200
# done



# graph="latin"
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, GI, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in {17..30}
# do
#     python main_GI.py --instance $instance --graph $graph --seed $seed --iters $iters --batch_size $batch_size --optimizer $optimizer
# done