import click
from omegaconf import OmegaConf

from showerpipe.generator import PythiaGenerator
from showerpipe.pipeline import sources, sinks, filters
from showerpipe.pipeline import factory, load
from showerpipe.pipeline import construct_pipeline


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
    ctx.obj = {}
    conf = OmegaConf.load(config_path)
    plugin_list = [] if conf.plugins == [None] else conf.plugins
    if plugin_list and plugin_list != [None]:
        load.load_plugins(plugin_list)
    factory.register('hdf_sink', sinks.HdfSink)
    factory.register('knn_filter', filters.KnnTransform)
    pipeline_tree = OmegaConf.to_object(conf.pipeline)
    if 'branch' not in pipeline_tree[0]:
        pipeline_tree = [{'branch': pipeline_tree}]
    ctx.obj['pipeline'] = construct_pipeline(pipeline_tree)


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
    shower_source = sources.ShowerSource(data_generator)
    run(shower_source, ctx.obj['pipeline'])
