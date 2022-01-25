import click
from omegaconf import OmegaConf

from showerpipe.generator import PythiaGenerator
from showerpipe.pipeline.sources import ShowerSource
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
    load.load_plugins(conf['plugins'])
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
    pipeline = ShowerSource(data_generator)
    run(pipeline, ctx.obj['listeners'])
