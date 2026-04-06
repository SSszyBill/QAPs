seed=42
# iters=5000
# batch_size=50
# optimizer="rmsprop"

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
for instance in $(cat qap_instances.txt); do
    python test_baseline.py --instance $instance 
done

# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat qaplib.txt); do
#     python main_gpu3.py --instance QAPLIB/$instance --seed $seed
# done

# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat qap_instances.txt); do
#     for dual_init in 100; do
#         for primal_lr in 0.02; do
#             for dual_lr in 0.02; do
#                 python main.py --instance $instance --seed $seed --primal_lr $primal_lr --dual_lr $dual_lr --dual_init $dual_init
#             done
#         done
#     done
# done


# seed=42
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# for solver in ngm; do
#     echo "==========Running pygm at $timestamp, seed $seed, instance $instance, solver $solver==========" >> results/pygm/${solver}.txt
#     for instance in $(cat qaplib.txt); do
#         python baselines/pygm.py --instance QAPLIB/$instance --solver $solver --seed $seed
#     done
# done





# seed=42
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat paley.txt); do
#     python main.py --instance GI/paley/$instance --seed $seed --dual_init 40
# done




# seed=42
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat latin.txt); do
#     python main.py --instance GI/latin/$instance --seed $seed --dual_init 20 --primal_lr 0.03
# done




# seed=47
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat qaplib.txt); do
#     python main_gpu3.py --instance QAPLIB/$instance --seed $seed
# done

# seed=1111
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat qaplib.txt); do
#     python main_gpu3.py --instance QAPLIB/$instance --seed $seed
# done

# seed=1234
# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat qaplib.txt); do
#     python main_gpu3.py --instance QAPLIB/$instance --seed $seed
# done

# timestamp=$(date +"%Y-%m-%d %H:%M:%S")
# echo "==========Running script at $timestamp, seed $seed, iters $iters, batch_size $batch_size, optimizer $optimizer==========" >> result.txt
# for instance in $(cat qap_instances.txt); do
#     python main_gpu3.py --instance $instance --seed $seed
# done

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




# # 最简单的方法：nohup + &
# nohup python your_script.py > output.log 2>&1 &

# # 或者将输出和错误分开
# nohup python your_script.py > output.log 2> error.log &

# # 如果不需要保存输出
# nohup python your_script.py > /dev/null 2>&1 &

# # 查看后台作业
# jobs -l

# # 重新连接到输出（实时查看）
# tail -f output.log

# kill xxxx