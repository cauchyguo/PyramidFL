
# Submit job to the remote cluster

import yaml
import sys
import time
import random
import os
import subprocess
import pickle
import datetime

# sys.path.append("..")
# from argParser import args

import socket


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def process_cmd(yaml_file):
    current_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(current_path, yaml_file)
    print(config_path)
    # 以字典方式载入conf配置信息
    yaml_conf = load_yaml_conf(config_path)

    # yaml_conf = load_yaml_conf(yaml_file)
    # ps_ip = yaml_conf['ps_ip']
    # ps ip为127.0.1.1
    ps_ip = socket.gethostname()
    worker_ips, total_gpus = [], []

    # 简化版，只使用一台服务器四个gpu进行仿真实验
    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        ip = socket.gethostname()
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))

    running_vms = set()
    # 子进程
    subprocess_list = set()
    # submit_user:ypguo@
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(
        yaml_conf['auth']['ssh_user']) else ""

    # 每个gpu开启一个进程（共3个）
    total_gpu_processes = sum([sum(x) for x in total_gpus])
    # 1-2-3
    learner_conf = '-'.join([str(_)
                            for _ in list(range(1, total_gpu_processes+1))])

    conf_script = ''
    setup_cmd = ''
    # 实际设置为空
    if yaml_conf['setup_commands'] is not None:
        for item in yaml_conf['setup_commands']:
            setup_cmd += (item + ' && ')

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        '%m%d_%H%M%S')+'_'+str(random.randint(1, 60000))

    time_to_print = datetime.datetime.strptime(
        '_'.join(time_stamp.split('_')[:2]), '%m%d_%H%M%S')
    time_to_print = time_to_print.strftime('%m/%d,%H:%M:%S')
    # 运行时配置
    job_conf = {'time_stamp': time_stamp,
                'total_worker': total_gpu_processes,
                'ps_ip': ps_ip,
                'ps_port': random.randint(1000, 60000),
                'manager_port': random.randint(1000, 60000),
                }
    # 预定义配置
    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    job_name = job_conf['job_name']
    if len(sys.argv) > 3:
        job_conf['sample_mode'] = sys.argv[3]
    if len(sys.argv) > 4:
        job_conf['load_model'] = True
        job_conf['load_time_stamp'] = sys.argv[4]
        job_conf['load_epoch'] = sys.argv[5]
        job_conf["model_path"] = os.path.join(
            job_conf["log_path"], 'logs', job_name, job_conf['load_time_stamp'])

    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'

    # 运行时生成的logging文件（改成放置在logging文件夹下）
    log_file_name = os.path.join(
        current_path, 'logging', f"{job_name}_logging_{time_stamp}")

    # =========== Submit job to parameter server ============

    running_vms.add(ps_ip)
    # conf_script为预定的和既定的job_conf中的配置
    # 通关传参的方式指定agg的运行配置
    ps_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} --gpu_device=0"

    print(f"Starting time_stamp on {time_stamp}...")

    with open(log_file_name, 'wb') as fout:
        pass

    training_history_keys = ['data_set',
                             'task',
                             'model',
                             'epochs',
                             'sample_mode',
                             #  'gradient_policy',
                             ]
    print("Begin at:" + time_to_print + '\n' + 'Job Conf:')

    for _i, key in enumerate(training_history_keys):
        if _i == 3:
            print('\t' + 'client_nums: ' + str(job_conf['total_worker']))
        print('\t' + key + ': ' + str(job_conf[key]))

    print(f"Starting aggregator on {ps_ip}...")
    with open(log_file_name, 'a') as fout:
        # p=subprocess.Popen(f'ssh -tt {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"', shell=True, stdout=fout, stderr=fout)

        # p=subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
        cmd_sequence = f'{ps_cmd}'
        cmd_sequence = cmd_sequence.split()
        # cmd_sequence即运行param_server
        p = subprocess.Popen(cmd_sequence, stdout=fout, stderr=fout)

        subprocess_list.add(p)
        # time.sleep(30)
        time.sleep(3)

    # =========== Submit job to each worker ============
    rank_id = 1  # rank表示gpu设备对应的id
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")
        # gpu:[0, 1, 1, 1]
        # 针对多服务器多卡，rankid为显卡标识
        for gpu_device in range(len(gpu)):
            for _ in range(gpu[gpu_device]):
                worker_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} --gpu_device={gpu_device}"

                print("worker" + str(rank_id) + ":gpu" +
                      str(gpu_device) + "; id: " + str(rank_id))
                rank_id += 1

                with open(log_file_name, 'a') as fout:
                    # p=subprocess.Popen(f'ssh -tt {submit_user}{worker} "{setup_cmd} {worker_cmd}"', shell=True, stdout=fout, stderr=fout)

                    # p=subprocess.Popen(f'{worker_cmd}', shell=True, stdout=fout, stderr=fout)

                    cmd_sequence = f'{worker_cmd}'
                    cmd_sequence = cmd_sequence.split()
                    # subprocess.Popen创建一个子进程（通过列表的方式），并连接到他的输入输出管道以获取返回值
                    p = subprocess.Popen(
                        cmd_sequence, stdout=fout, stderr=fout)

                    subprocess_list.add(p)

    exit_codes = [p.wait() for p in subprocess_list]

    # dump the address of running workers
    job_name = os.path.join(current_path, "user_vms_log",
                            f"{job_name}_{time_stamp}")
    with open(job_name, 'wb') as fout:
        job_meta = {'user': submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(
        f"Submitted job, please check your logs ({log_file_name}) for status")


def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
        print(f"Shutting down job on {vm_ip}")
        os.system(
            f"ssh {job_meta['user']}{vm_ip} '/home/ypguo/.conda/envs/fedscale/bin/python {current_path}/shutdown.py {job_name}'")


try:
    if len(sys.argv) == 1:
        # process_cmd('configs/har/conf_test.yml')
        # process_cmd('configs/openimage/conf_test.yml')
        process_cmd('configs/speech/conf_test.yml')
        # process_cmd('configs/stackoverflow/conf_test.yml')
    elif sys.argv[1] == 'submit':
        process_cmd(sys.argv[2])
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    else:
        print("Unknown cmds ...")
except Exception as e:
    print(f"====Error {e.args}")
