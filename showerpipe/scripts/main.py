"""
Parallel h5py with MPI
======================

"""
import tempfile
from copy import deepcopy

import numpy as np
from mpi4py import MPI
import h5py

from showerpipe import lhe
from showerpipe.generator import PythiaGenerator
from typicle import Types


def create_dset(buffer, name, shape, dtype):
    return buffer.create_dataset(name, shape, dtype=dtype)

def main(lhe_path, comm, root=0, record_length=1000):
    # setup comm info:
    type_ = Types()
    proc_name = MPI.Get_processor_name()
    rank = int(comm.Get_rank())
    num_procs = comm.Get_size()
    # open the file (examples don't use with - why not?)
    f = h5py.File('test/big_test.hdf5', 'w', driver='mpio', comm=comm)
    process = f.create_group('process')
    # prepare data
    hard_events = lhe.load_lhe(lhe_path)
    hard_events.tile(repeats=10, inplace=True)
    num_events = hard_events.num_events
    records = {
            key: []
            for key in ('count', 'edges', 'pmu', 'pdg', 'color', 'final')
            }
    flush_num = 0
    with tempfile.NamedTemporaryFile() as lhe_temp:
        lhe_temp.write(hard_events.content)
        lhe_temp.seek(0)
        pythia_generator = PythiaGenerator(
                config_file='shower_settings.cmnd',
                me_file=lhe_temp.name,
                rng_seed=rank,
                )
        for event_num, event in enumerate(pythia_generator):
            records['count'].append(event.count)
            records['edges'].append(event.edges)
            records['pmu'].append(event.pmu)
            records['pdg'].append(event.pdg)
            records['color'].append(event.color)
            records['final'].append(event.final)
            if ( (event_num + 1) % record_length == 0
                 or (event_num + 1) == num_events):
                records['count'].reverse() # popping order from the stack
                send_counts = np.array(records['count'], dtype='<i4')
                recv_counts = np.zeros(record_length * num_procs, dtype='<i4')
                comm.Allgather(sendbuf=send_counts, recvbuf=recv_counts)
                for count_idx, num_pcls in enumerate(recv_counts):
                    idx = num_procs * record_length * flush_num + count_idx
                    grp = process.create_group(f'event_{idx:09}')
                    grp.attrs['num_pcls'] = num_pcls
                    pmu = create_dset(grp, 'pmu', (num_pcls,), type_.pmu)
                    pdg = create_dset(grp, 'pdg', (num_pcls,), type_.pdg)
                    final = create_dset(grp, 'final', (num_pcls,), type_.final)
                    color = create_dset(grp, 'color', (num_pcls,), type_.color)
                    edges = create_dset(grp, 'edges', (num_pcls,), type_.edge)
                for count_idx, count in enumerate(records['count']):
                    idx = int(count_idx
                              + rank * record_length
                              + num_procs * record_length * flush_num
                              )
                    curr_grp = process[f'event_{idx:09}']
                    curr_grp['pmu'][...] = records['pmu'].pop()
                    curr_grp['pdg'][...] = records['pdg'].pop()
                    curr_grp['final'][...] = records['final'].pop()
                    curr_grp['color'][...] = records['color'].pop()
                    curr_grp['edges'][...] = records['edges'].pop()
                flush_num += 1
                records['count'] = []
    f.close()

# if __name__ == '__main__':
#     lhe_path = str('/scratch/jlc1n20/data/g_to_bb/Events/run_01/'
#                    + 'unweighted_events.lhe.gz')
#     comm = MPI.COMM_WORLD
#     main(lhe_path=lhe_path, comm=comm)
