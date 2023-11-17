from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, lstm_benchmark, utils
from copy import copy

@utils.debug
def Compare(args):
    results = []
    num_iters = 1000
    args.gpu = True
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        for batch_size in [64, 128, 256]:
            for seq_length in [20, 100]:
                for hidden_dim in [40, 100, 400, 800]:
                    args.batch_size = batch_size
                    args.seq_length = seq_length
                    args.hidden_dim = hidden_dim
                    args.data_size = batch_size * seq_length * num_iters
                    args.iters_to_report = num_iters // 3

                    args.implementation = 'own'
                    t_own = lstm_benchmark.Benchmark(args)
                    workspace.ResetWorkspace()
                    args.implementation = 'cudnn'
                    t_cudnn = lstm_benchmark.Benchmark(args)
                    workspace.ResetWorkspace()
                    results.append((copy(args), float(t_own), float(t_cudnn)))
                    print(args)
                    print(f"t_cudnn / t_own: {t_cudnn / t_own}")

    for args, t_own, t_cudnn in results:
        print(
            f"{str(args)}: cudnn time: {t_cudnn}, own time: {t_own}, ratio: {t_cudnn / t_own}"
        )

    ratio_sum = 0
    for args, t_own, t_cudnn in results:
        ratio = float(t_cudnn) / t_own
        ratio_sum += ratio
        print(
            f"hidden_dim: {args.hidden_dim}, seq_lengths: {args.seq_length}, batch_size: {args.batch_size}, num_layers: {args.num_layers}: cudnn time: {t_cudnn}, own time: {t_own}, ratio: {ratio}"
        )

    print(f"Ratio average: {ratio_sum / len(results)}")


if __name__ == '__main__':
    args = lstm_benchmark.GetArgumentParser().parse_args()

    workspace.GlobalInit([
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_print_blob_sizes_at_exit=0',
        '--caffe2_gpu_memory_tracking=1'])

    Compare(args)
