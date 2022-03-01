import io
from contextlib import redirect_stdout
import click
from omegaconf import OmegaConf
from math import ceil

from showerpipe.generator import PythiaGenerator
from showerpipe.pipeline import sources, sinks, filters
from showerpipe.pipeline import factory, load
from showerpipe.pipeline import construct_pipeline
from showerpipe.lhe import LheData, split, count_events
from showerpipe._version import version


def run(shower_source, pipeline):
    """Attaches the pipeline to the event generator, and then runs it."""
    shower_source.attach(pipeline)
    with click.progressbar(shower_source) as events:
        for event in events:
            event.notify()


@click.group()
@click.argument('config_path', type=click.Path(exists=True))
@click.pass_context
def showergen(ctx, config_path):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()

    if rank == 0:
        title = 'Showerpipe Generator'
        underline = '-' * len(title)
        version_str = f'version: {version}'
        title_fmt = click.style(title, fg='green')
        title_header = f'\n{title_fmt}\n{underline}\n{version_str}\n'
        click.echo(title_header)

    ctx.obj = {}
    ctx.obj['comm'] = comm

    hide_stdout = io.StringIO()
    with redirect_stdout(hide_stdout):
        conf = OmegaConf.load(config_path)
    plugin_list = [] if conf.plugins == [None] else conf.plugins
    if plugin_list and plugin_list != [None]:
        load.load_plugins(plugin_list)
    factory.register('hdf_sink', sinks.HdfSink)
    factory.register('knn_filter', filters.KnnTransform)
    with redirect_stdout(hide_stdout):
        pipeline_tree = OmegaConf.to_object(conf.pipeline)
    if 'branch' not in pipeline_tree[0]:
        pipeline_tree = [{'branch': pipeline_tree}]
    ctx.obj['pipeline'] = construct_pipeline(pipeline_tree, rank)


@showergen.command()
@click.argument('settings-file', type=click.Path(exists=True))
@click.option(
        '--me-file', type=click.Path(exists=True),
        help='Les Houches file containing hard events.'
        )
@click.pass_context
def pythia(ctx, settings_file, me_file):
    """Runs Pythia using SETTINGS_FILE, typically with extension 'cmnd'."""
    comm = ctx.obj['comm']
    num_procs: int = comm.Get_size()
    rank: int = comm.Get_rank()

    if rank == 0:  # split up the lhe file
        pythia_name = click.style('Pythia8', fg='red')
        num_events = count_events(me_file)
        header = f'Using {pythia_name} to shower and hadronise {num_events} '
        header += f'events with {num_procs} cores.'
        click.echo(header)

        stride = ceil(num_events / num_procs)
        lhe_splits = split(me_file, stride)
        data = next(lhe_splits)
        for i in range(1, num_procs):
            comm.send(next(lhe_splits), dest=i, tag=10+i)
    else:
        data = comm.recv(source=0, tag=10+rank)

    gen = PythiaGenerator(settings_file, data)
    shower_source = sources.ShowerSource(gen)
    run(shower_source, ctx.obj['pipeline'])
