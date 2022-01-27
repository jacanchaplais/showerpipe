import click
from omegaconf import OmegaConf

from showerpipe.generator import PythiaGenerator
from showerpipe.pipeline import sources, sinks, filters
from showerpipe.pipeline import factory, load


def run(pipeline, listeners):
    """Attaches the observers to the event generator, and then runs it."""
    for listen in listeners:
        pipeline.attach(listen)
    with click.progressbar(pipeline) as events:
        for event in events:
            event.notify()


@click.group()
@click.argument('config_path', type=click.Path(exists=True))
@click.pass_context
def showergen(ctx, config_path):
    ctx.obj = {}
    conf = OmegaConf.load(config_path)
    plugin_list = conf['plugins']
    plugin_list = [] if plugin_list == [None] else plugin_list
    if plugin_list and plugin_list != [None]:
        load.load_plugins(plugin_list)
    factory.register('hdf_sink', sinks.HdfSink)
    factory.register('knn_filter', filters.KnnTransform)
    ctx.obj['listeners'] = [factory.create(item) for item in conf['pipeline']]


@showergen.command()
@click.argument('settings-file', type=click.Path(exists=True))
@click.option(
        '--me-file', type=click.Path(exists=True),
        help='Les Houches file containing hard events.'
        )
@click.pass_context
def pythia(ctx, settings_file, me_file):
    """Runs Pythia using SETTINGS_FILE, typically with extension 'cmnd'."""
    data_generator = PythiaGenerator(
            config_file=settings_file,
            me_file=me_file,
            )
    pipeline = sources.ShowerSource(data_generator)
    run(pipeline, ctx.obj['listeners'])
